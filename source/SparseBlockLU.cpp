#include <cuda_runtime.h>
#include <thrust/complex.h>
#include <thrust/device_vector.h>

#include "SparseMatrix.hpp"
#include "forall.hpp"

#ifdef CUDDH_HAS_CUDSS
#include <cudss.h>
#endif

namespace cuddh
{

    // ─── CuDSS helpers ───────────────────────────────────────────────────────────

#ifdef CUDDH_HAS_CUDSS

    static void cudss_check(cudssStatus_t status, const char *op)
    {
        if (status != CUDSS_STATUS_SUCCESS)
            throw std::runtime_error(std::string("SparseBlockLU: CuDSS error in ") + op +
                                     " (status=" + std::to_string(static_cast<int>(status)) + ")");
    }

    template <typename value_t>
    constexpr cudaDataType_t cudss_value_type();
    template <>
    constexpr cudaDataType_t cudss_value_type<float>()
    {
        return CUDA_R_32F;
    }
    template <>
    constexpr cudaDataType_t cudss_value_type<double>()
    {
        return CUDA_R_64F;
    }
    template <>
    constexpr cudaDataType_t cudss_value_type<std::complex<float>>()
    {
        return CUDA_C_32F;
    }
    template <>
    constexpr cudaDataType_t cudss_value_type<std::complex<double>>()
    {
        return CUDA_C_64F;
    }

    // Blocked-complex device buffer → interleaved complex device buffer.
    // d_blocked layout: block p starts at p * 2 * max_n:
    //   re_0..re_{n_p-1} at [p*2*max_n .. p*2*max_n + n_p - 1]
    //   im_0..im_{n_p-1} at [p*2*max_n + max_n .. p*2*max_n + max_n + n_p - 1]
    // d_cx layout: interleaved, block p starts at d_offsets[p] (prefix sum of n_p).
    template <typename scalar_t>
    static void blocked_to_cx(const scalar_t *d_blocked, thrust::complex<scalar_t> *d_cx, int n_blocks, int max_n,
                              const int *d_sizes, const int *d_offsets)
    {
        forall(n_blocks * max_n, [=] __device__(int tid) {
            const int p = tid / max_n;
            const int i = tid % max_n;
            if (i >= d_sizes[p])
                return;
            const scalar_t re = d_blocked[p * 2 * max_n + i];
            const scalar_t im = d_blocked[p * 2 * max_n + max_n + i];
            d_cx[d_offsets[p] + i] = thrust::complex<scalar_t>(re, im);
        });
    }

    // Interleaved complex device buffer → blocked-complex device buffer.
    template <typename scalar_t>
    static void cx_to_blocked(const thrust::complex<scalar_t> *d_cx, scalar_t *d_blocked, int n_blocks, int max_n,
                              const int *d_sizes, const int *d_offsets)
    {
        forall(n_blocks * max_n, [=] __device__(int tid) {
            const int p = tid / max_n;
            const int i = tid % max_n;
            if (i >= d_sizes[p])
                return;
            const thrust::complex<scalar_t> v = d_cx[d_offsets[p] + i];
            d_blocked[p * 2 * max_n + i] = v.real();
            d_blocked[p * 2 * max_n + max_n + i] = v.imag();
        });
    }

#endif // CUDDH_HAS_CUDSS

    // ─── Impl ────────────────────────────────────────────────────────────────────

    template <typename scalar_t, bool Complex>
    struct SparseBlockLU<scalar_t, Complex>::Impl
    {
        using value_t = typename SparseBlockLU::value_t;

        SparseBlockLUStats<scalar_t, Complex> stats;

#ifdef CUDDH_HAS_CUDSS
        cudssHandle_t handle{nullptr};
        cudssConfig_t config{nullptr};
        cudssData_t data{nullptr};
        cudssMatrix_t A_matrix{nullptr};
        cudssMatrix_t x_matrix{nullptr};
        cudssMatrix_t b_matrix{nullptr};

        // Owned device copies of all blocks' CSR data (concatenated).
        // Block p's row pointers start at all_row_ptrs+rp_offsets[p] (length n_p + 1).
        // Block p's column indices start at all_cols+nz_offsets[p] (length nnz_p).
        // Block p's values start at all_vals+nz_offsets[p].
        // Using raw cudaMalloc to match the CuDSS sample pattern exactly.
        int *all_row_ptrs{nullptr};
        int *all_cols{nullptr};
        value_t *all_vals{nullptr};

        // Per-block sizes (n_p) and DOF prefix sums (for locating each block in d_rhs_cx / d_x_cx).
        // d_dof_offsets[p] = sum(n_i for i < p).
        thrust::device_vector<int> d_sizes;
        thrust::device_vector<int> d_dof_offsets;

        // Host metadata arrays kept alive for the entire lifetime of the CuDSS batch matrices.
        // CuDSS stores POINTERS to these arrays (not copies) when CUDA_R_32I index type is used,
        // so they must not be destroyed while A_matrix / b_matrix / x_matrix are alive.
        std::vector<int> h_nrows_v, h_ncols_v, h_nnz_v, h_ld_v, h_ncols_dn_v;

        // Interleaved complex RHS and solution buffers (device). Stored in one contiguous
        // allocation: [rhs_0..rhs_{n-1}, x_0..x_{n-1}] where n = total_dof.
        thrust::complex<scalar_t> *d_cx_buf{nullptr}; // raw device allocation, size = 2*total_dof
        thrust::complex<scalar_t> *d_rhs_cx{nullptr}; // = d_cx_buf
        thrust::complex<scalar_t> *d_x_cx{nullptr};   // = d_cx_buf + total_dof

        // Device-side arrays of device pointers — required by the CuDSS batch API.
        // Allocated via cudaMalloc (not thrust) to match the CuDSS sample exactly.
        void **d_rowstart_arr{nullptr};
        void **d_col_arr{nullptr};
        void **d_val_arr{nullptr};
        void **d_rhs_arr{nullptr};
        void **d_x_arr{nullptr};

        ~Impl()
        {
            if (x_matrix)
                cudssMatrixDestroy(x_matrix);
            if (b_matrix)
                cudssMatrixDestroy(b_matrix);
            if (A_matrix)
                cudssMatrixDestroy(A_matrix);
            if (data)
                cudssDataDestroy(handle, data);
            if (config)
                cudssConfigDestroy(config);
            if (handle)
                cudssDestroy(handle);
            cudaFree(d_rowstart_arr);
            cudaFree(d_col_arr);
            cudaFree(d_val_arr);
            cudaFree(d_rhs_arr);
            cudaFree(d_x_arr);
            cudaFree(all_row_ptrs);
            cudaFree(all_cols);
            cudaFree(all_vals);
            cudaFree(d_cx_buf);
        }
#else
        ~Impl() = default;
#endif
    };

    // ─── Constructor ─────────────────────────────────────────────────────────────

    template <typename scalar_t, bool Complex>
    SparseBlockLU<scalar_t, Complex>::SparseBlockLU(const BlockSparseMatrix<scalar_t, Complex> &blocks)
        : _pimpl(std::make_unique<Impl>())
    {
        static_assert(Complex, "SparseBlockLU: only the complex (Complex=true) specialization is implemented");

        const int nb = blocks.n_blocks();
        if (nb == 0)
            throw std::logic_error("SparseBlockLU: BlockSparseMatrix has no blocks");
        if (blocks.state() != SparseMatrixState::Finalized)
            throw std::logic_error("SparseBlockLU: BlockSparseMatrix is not finalized");

        auto &p = *_pimpl;
        p.stats.n_blocks = nb;

#ifdef CUDDH_HAS_CUDSS
        // ── Gather per-block metadata ──────────────────────────────────────────────
        // Stored in Impl so they outlive the constructor.
        // CuDSS holds host pointers to these arrays (not copies) when CUDA_R_32I is used.
        p.h_nrows_v.resize(nb);
        p.h_ncols_v.resize(nb);
        p.h_nnz_v.resize(nb);
        p.h_ld_v.resize(nb);
        p.h_ncols_dn_v.assign(nb, 1);
        auto &h_nrows = p.h_nrows_v;
        auto &h_ncols = p.h_ncols_v;
        auto &h_nnz = p.h_nnz_v;
        auto &h_ld = p.h_ld_v;
        auto &h_ncols_dn = p.h_ncols_dn_v;

        std::vector<int> h_sizes(nb);
        std::vector<int> h_dof_offsets(nb + 1, 0);

        int total_dof = 0;
        int max_n = 0;

        for (int b = 0; b < nb; ++b)
        {
            const int n = blocks.block_size(b);
            const int nnz = blocks.block_nnz(b);

            h_nrows[b] = n;
            h_ncols[b] = n;
            h_nnz[b] = nnz;
            h_ld[b] = n;
            h_sizes[b] = n;
            h_dof_offsets[b + 1] = h_dof_offsets[b] + n;

            total_dof += n;
            max_n = std::max(max_n, n);
            p.stats.total_nnz += nnz;
        }
        p.stats.max_n = max_n;

        // ── Copy flat CSR data to device in three single transfers ─────────────────
        const int total_rp = blocks.rp_offset(nb);
        const int total_nnz = blocks.total_nnz();

        cudaMalloc(&p.all_row_ptrs, static_cast<size_t>(total_rp) * sizeof(int));
        cudaMalloc(&p.all_cols, static_cast<size_t>(total_nnz) * sizeof(int));
        cudaMalloc(&p.all_vals, static_cast<size_t>(total_nnz) * sizeof(value_t));

        cudaMemcpy(p.all_row_ptrs, blocks.row_ptrs(), static_cast<size_t>(total_rp) * sizeof(int),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(p.all_cols, blocks.col_indices(), static_cast<size_t>(total_nnz) * sizeof(int),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(p.all_vals, blocks.values(), static_cast<size_t>(total_nnz) * sizeof(value_t),
                   cudaMemcpyHostToDevice);

        // RHS and solution in one contiguous allocation.
        cudaMalloc(&p.d_cx_buf, 2 * static_cast<size_t>(total_dof) * sizeof(thrust::complex<scalar_t>));
        cudaMemset(p.d_cx_buf, 0, 2 * static_cast<size_t>(total_dof) * sizeof(thrust::complex<scalar_t>));
        p.d_rhs_cx = p.d_cx_buf;
        p.d_x_cx = p.d_cx_buf + total_dof;

        // Upload per-block size arrays to device (used by the conversion kernels).
        p.d_sizes.assign(h_sizes.begin(), h_sizes.end());
        p.d_dof_offsets.assign(h_dof_offsets.begin(), h_dof_offsets.begin() + nb);

        // ── Build device arrays of device pointers for the batch descriptors ──────
        {
            std::vector<void *> h_rowstart_arr(nb), h_col_arr(nb), h_val_arr(nb);
            std::vector<void *> h_rhs_arr(nb), h_x_arr(nb);

            thrust::complex<scalar_t> *rhs_base = p.d_rhs_cx;
            thrust::complex<scalar_t> *x_base = p.d_x_cx;

            for (int b = 0; b < nb; ++b)
            {
                h_rowstart_arr[b] = p.all_row_ptrs + blocks.rp_offset(b);
                h_col_arr[b] = p.all_cols + blocks.nz_offset(b);
                h_val_arr[b] = p.all_vals + blocks.nz_offset(b);
                h_rhs_arr[b] = rhs_base + h_dof_offsets[b];
                h_x_arr[b] = x_base + h_dof_offsets[b];
            }

            const size_t ptr_bytes = static_cast<size_t>(nb) * sizeof(void *);
            cudaMalloc(&p.d_rowstart_arr, ptr_bytes);
            cudaMalloc(&p.d_col_arr, ptr_bytes);
            cudaMalloc(&p.d_val_arr, ptr_bytes);
            cudaMalloc(&p.d_rhs_arr, ptr_bytes);
            cudaMalloc(&p.d_x_arr, ptr_bytes);

            cudaMemcpy(p.d_rowstart_arr, h_rowstart_arr.data(), ptr_bytes, cudaMemcpyHostToDevice);
            cudaMemcpy(p.d_col_arr, h_col_arr.data(), ptr_bytes, cudaMemcpyHostToDevice);
            cudaMemcpy(p.d_val_arr, h_val_arr.data(), ptr_bytes, cudaMemcpyHostToDevice);
            cudaMemcpy(p.d_rhs_arr, h_rhs_arr.data(), ptr_bytes, cudaMemcpyHostToDevice);
            cudaMemcpy(p.d_x_arr, h_x_arr.data(), ptr_bytes, cudaMemcpyHostToDevice);
        }

        // ── Create CuDSS objects ───────────────────────────────────────────────────
        cudss_check(cudssCreate(&p.handle), "cudssCreate");
        cudss_check(cudssConfigCreate(&p.config), "cudssConfigCreate");
        cudss_check(cudssDataCreate(p.handle, &p.data), "cudssDataCreate");

        cudss_check(cudssMatrixCreateBatchCsr(&p.A_matrix, static_cast<int64_t>(nb), h_nrows.data(), h_ncols.data(),
                                              h_nnz.data(), p.d_rowstart_arr, nullptr, p.d_col_arr, p.d_val_arr,
                                              CUDA_R_32I, cudss_value_type<value_t>(), CUDSS_MTYPE_GENERAL,
                                              CUDSS_MVIEW_FULL, CUDSS_BASE_ZERO),
                    "cudssMatrixCreateBatchCsr");

        cudss_check(cudssMatrixCreateBatchDn(&p.b_matrix, static_cast<int64_t>(nb), h_nrows.data(), h_ncols_dn.data(),
                                             h_ld.data(), p.d_rhs_arr, CUDA_R_32I, cudss_value_type<value_t>(),
                                             CUDSS_LAYOUT_COL_MAJOR),
                    "cudssMatrixCreateBatchDn (b)");

        cudss_check(cudssMatrixCreateBatchDn(&p.x_matrix, static_cast<int64_t>(nb), h_nrows.data(), h_ncols_dn.data(),
                                             h_ld.data(), p.d_x_arr, CUDA_R_32I, cudss_value_type<value_t>(),
                                             CUDSS_LAYOUT_COL_MAJOR),
                    "cudssMatrixCreateBatchDn (x)");

        // ── Analysis ───────────────────────────────────────────────────────────────
        const auto t_a0 = std::chrono::high_resolution_clock::now();
        cudss_check(cudssExecute(p.handle, CUDSS_PHASE_ANALYSIS, p.config, p.data, p.A_matrix, p.x_matrix, p.b_matrix),
                    "cudssExecute(ANALYSIS)");
        if (cudaError_t e = cudaDeviceSynchronize(); e != cudaSuccess)
            throw std::runtime_error(std::string("SparseBlockLU: ANALYSIS CUDA error: ") + cudaGetErrorString(e));
        p.stats.analysis_seconds =
            std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t_a0).count();

        // ── Factorization ──────────────────────────────────────────────────────────
        const auto t_f0 = std::chrono::high_resolution_clock::now();
        cudss_check(
            cudssExecute(p.handle, CUDSS_PHASE_FACTORIZATION, p.config, p.data, p.A_matrix, p.x_matrix, p.b_matrix),
            "cudssExecute(FACTORIZATION)");
        if (cudaError_t e = cudaDeviceSynchronize(); e != cudaSuccess)
            throw std::runtime_error(std::string("SparseBlockLU: FACTORIZATION CUDA error: ") + cudaGetErrorString(e));
        p.stats.factor_seconds =
            std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t_f0).count();

        int64_t lu_nnz = 0;
        cudssDataGet(p.handle, p.data, CUDSS_DATA_LU_NNZ, &lu_nnz, sizeof(lu_nnz), nullptr);
        p.stats.factor_bytes = (lu_nnz > 0) ? static_cast<size_t>(lu_nnz) * sizeof(value_t)
                                            : static_cast<size_t>(total_nnz) * sizeof(value_t);

#else
        throw std::logic_error("SparseBlockLU: CuDSS is required. Rebuild with -DCUDDH_USE_CUDSS=ON.");
#endif
    }

    // ─── Destructor / Move ────────────────────────────────────────────────────────

    template <typename scalar_t, bool Complex>
    SparseBlockLU<scalar_t, Complex>::~SparseBlockLU() = default;

    template <typename scalar_t, bool Complex>
    SparseBlockLU<scalar_t, Complex>::SparseBlockLU(SparseBlockLU &&) noexcept = default;

    template <typename scalar_t, bool Complex>
    SparseBlockLU<scalar_t, Complex> &SparseBlockLU<scalar_t, Complex>::operator=(SparseBlockLU &&) noexcept = default;

    // ─── stats / print ────────────────────────────────────────────────────────────

    template <typename scalar_t, bool Complex>
    const SparseBlockLUStats<scalar_t, Complex> &SparseBlockLU<scalar_t, Complex>::stats() const
    {
        return _pimpl->stats;
    }

    template <typename scalar_t, bool Complex>
    void SparseBlockLU<scalar_t, Complex>::print(std::ostream &os) const
    {
        const auto &s = _pimpl->stats;
        os << "SparseBlockLU(n_blocks=" << s.n_blocks << ", max_n=" << s.max_n << ", total_nnz=" << s.total_nnz << ")\n"
           << "  analysis: " << s.analysis_seconds << " s\n"
           << "  factor:   " << s.factor_seconds << " s\n"
           << "  solves:   " << s.solve_calls << " calls, " << s.total_solve_seconds << " s total\n"
           << "  memory:   factor=" << s.factor_bytes << " B\n";
    }

    // ─── solve ────────────────────────────────────────────────────────────────────

    template <typename scalar_t, bool Complex>
    bool SparseBlockLU<scalar_t, Complex>::solve(const scalar_t *d_rhs, scalar_t *d_x)
    {
#ifdef CUDDH_HAS_CUDSS
        auto &p = *_pimpl;

        const auto t0 = std::chrono::high_resolution_clock::now();

        const int nb = p.stats.n_blocks;
        const int max_n = p.stats.max_n;
        const int *d_sizes = thrust::raw_pointer_cast(p.d_sizes.data());
        const int *d_offsets = thrust::raw_pointer_cast(p.d_dof_offsets.data());

        // 1. Convert blocked-complex input to per-block interleaved complex.
        blocked_to_cx(d_rhs, p.d_rhs_cx, nb, max_n, d_sizes, d_offsets);

        // 2. CuDSS batch solve: A_p * x_p = rhs_p for all p.
        cudssStatus_t st =
            cudssExecute(p.handle, CUDSS_PHASE_SOLVE, p.config, p.data, p.A_matrix, p.x_matrix, p.b_matrix);
        cudaDeviceSynchronize();

        if (st != CUDSS_STATUS_SUCCESS)
            throw std::runtime_error(
                "SparseBlockLU::solve: CuDSS batch solve failed (status=" + std::to_string(static_cast<int>(st)) + ")");

        // 3. Convert interleaved complex solution back to blocked format.
        cx_to_blocked(p.d_x_cx, d_x, nb, max_n, d_sizes, d_offsets);

        const auto t1 = std::chrono::high_resolution_clock::now();
        p.stats.total_solve_seconds += std::chrono::duration<double>(t1 - t0).count();
        ++p.stats.solve_calls;

        return true;
#else
        (void)d_rhs;
        (void)d_x;
        throw std::logic_error("SparseBlockLU: CuDSS is required. Rebuild with -DCUDDH_USE_CUDSS=ON.");
#endif
    }

    // ─── Explicit instantiations ──────────────────────────────────────────────────

    template class SparseBlockLU<float, true>;
    template class SparseBlockLU<double, true>;

} // namespace cuddh
