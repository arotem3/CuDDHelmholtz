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
            throw std::runtime_error(std::string("SparseLU: CuDSS error in ") + op +
                                     " (status=" + std::to_string(static_cast<int>(status)) + ")");
    }

    // cudaDataType_t for the matrix/vector value type
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

    static cudssMatrixType_t to_cudss_matrix_type(SparseMatrixType t)
    {
        switch (t)
        {
            case SparseMatrixType::Symmetric: return CUDSS_MTYPE_SYMMETRIC;
            case SparseMatrixType::Hermitian: return CUDSS_MTYPE_HERMITIAN;
            case SparseMatrixType::SPD:       return CUDSS_MTYPE_SPD;
            case SparseMatrixType::HPD:       return CUDSS_MTYPE_HPD;
            default:                          return CUDSS_MTYPE_GENERAL;
        }
    }

#endif // CUDDH_HAS_CUDSS

    // ─── Impl ────────────────────────────────────────────────────────────────────

    template <typename scalar_t, bool Complex>
    struct SparseLU<scalar_t, Complex>::Impl
    {
        using value_t = typename SparseLU::value_t;

        int n_rows{0};
        int n_cols{0};
        SparseLUStats<scalar_t, Complex> stats;

#ifdef CUDDH_HAS_CUDSS
        cudssHandle_t handle{nullptr};
        cudssConfig_t config{nullptr};
        cudssData_t data{nullptr};
        cudssMatrix_t A_matrix{nullptr};
        cudssMatrix_t x_matrix{nullptr};
        cudssMatrix_t b_matrix{nullptr};

        // Device CSR storage — CuDSS holds raw device pointers into these
        thrust::device_vector<int> csr_row_ptr;
        thrust::device_vector<int> csr_cols;
        thrust::device_vector<value_t> csr_vals;

        // Pre-allocated device vectors for RHS and solution (reused across solve calls)
        thrust::device_vector<value_t> x_dev;
        thrust::device_vector<value_t> b_dev;

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
        }

        void init_cudss()
        {
            cudss_check(cudssCreate(&handle), "cudssCreate");
            cudss_check(cudssConfigCreate(&config), "cudssConfigCreate");
            cudss_check(cudssDataCreate(handle, &data), "cudssDataCreate");
        }

        void build_matrix_descriptor(int n, int nnz, SparseMatrixType mat_type)
        {
            // rowStart = csr_row_ptr.data()   (first n of the n+1 entry row-ptr)
            // rowEnd   = csr_row_ptr.data()+1 (elements 1..n)
            int *rp = thrust::raw_pointer_cast(csr_row_ptr.data());
            int *ci = thrust::raw_pointer_cast(csr_cols.data());
            void *cv = thrust::raw_pointer_cast(csr_vals.data());

            // rowEnd = NULL tells CuDSS to interpret rowStart as a standard
            // n+1-entry CSR row-pointer array.
            cudss_check(cudssMatrixCreateCsr(&A_matrix, static_cast<int64_t>(n), static_cast<int64_t>(n),
                                             static_cast<int64_t>(nnz), rp, nullptr, ci, cv, CUDA_R_32I,
                                             cudss_value_type<value_t>(),
                                             to_cudss_matrix_type(mat_type), CUDSS_MVIEW_FULL,
                                             CUDSS_BASE_ZERO),
                        "cudssMatrixCreateCsr");
        }

        void build_rhs_descriptors(int n)
        {
            x_dev.assign(n, value_t{});
            b_dev.assign(n, value_t{});

            void *xp = thrust::raw_pointer_cast(x_dev.data());
            void *bp = thrust::raw_pointer_cast(b_dev.data());

            cudss_check(cudssMatrixCreateDn(&x_matrix, static_cast<int64_t>(n), 1, static_cast<int64_t>(n), xp,
                                            cudss_value_type<value_t>(), CUDSS_LAYOUT_COL_MAJOR),
                        "cudssMatrixCreateDn (x)");

            cudss_check(cudssMatrixCreateDn(&b_matrix, static_cast<int64_t>(n), 1, static_cast<int64_t>(n), bp,
                                            cudss_value_type<value_t>(), CUDSS_LAYOUT_COL_MAJOR),
                        "cudssMatrixCreateDn (b)");
        }
#else
        // Placeholder when CuDSS is not available
        ~Impl() = default;
#endif
    };

    // ─── Constructor / Destructor / Move ─────────────────────────────────────────

    template <typename scalar_t, bool Complex>
    SparseLU<scalar_t, Complex>::SparseLU(const SparseMatrix<scalar_t, Complex> &A) : _pimpl(std::make_unique<Impl>())
    {
        if (A.state() != SparseMatrixState::Finalized || !A._finalized_storage_present)
            throw std::logic_error("SparseLU: input matrix must be finalized before factorization");

        auto &p = *_pimpl;
        p.n_rows = A._rows;
        p.n_cols = A._cols;
        p.stats.n_rows = A._rows;
        p.stats.n_cols = A._cols;
        p.stats.nnz = A._nnz;
        p.stats.finalized_bytes = A.finalized_storage_bytes();

#ifdef CUDDH_HAS_CUDSS
        if (p.n_rows != p.n_cols)
            throw std::logic_error("SparseLU: factorization requires a square matrix");

        const int n = p.n_rows;
        const int nnz = A._nnz;

        // ── Analysis phase ──────────────────────────────────────────────────────
        const auto t_a0 = std::chrono::high_resolution_clock::now();

        p.init_cudss();

        // Copy CSR data to device
        {
            const int *rp = A._csr_row_ptr.host_read();
            p.csr_row_ptr.assign(rp, rp + n + 1);

            const int *ci = A._csr_cols.host_read();
            p.csr_cols.assign(ci, ci + nnz);

            const value_t *cv = A._csr_vals.host_read();
            p.csr_vals.assign(cv, cv + nnz);
        }

        p.build_matrix_descriptor(n, nnz, A.matrix_type());
        p.build_rhs_descriptors(n);

        cudss_check(cudssExecute(p.handle, CUDSS_PHASE_ANALYSIS, p.config, p.data, p.A_matrix, p.x_matrix, p.b_matrix),
                    "cudssExecute(ANALYSIS)");
        cudaDeviceSynchronize();

        const auto t_a1 = std::chrono::high_resolution_clock::now();
        p.stats.analysis_seconds = std::chrono::duration<double>(t_a1 - t_a0).count();

        // ── Factorization phase ─────────────────────────────────────────────────
        const auto t_f0 = std::chrono::high_resolution_clock::now();

        cudss_check(
            cudssExecute(p.handle, CUDSS_PHASE_FACTORIZATION, p.config, p.data, p.A_matrix, p.x_matrix, p.b_matrix),
            "cudssExecute(FACTORIZATION)");
        cudaDeviceSynchronize();

        const auto t_f1 = std::chrono::high_resolution_clock::now();
        p.stats.factor_seconds = std::chrono::duration<double>(t_f1 - t_f0).count();

        // Report factor memory from CuDSS
        int64_t lu_nnz = 0;
        cudssDataGet(p.handle, p.data, CUDSS_DATA_LU_NNZ, &lu_nnz, sizeof(lu_nnz), nullptr);
        p.stats.factor_bytes = static_cast<size_t>(lu_nnz) * sizeof(value_t);
        if (p.stats.factor_bytes == 0)
            p.stats.factor_bytes = static_cast<size_t>(nnz) * sizeof(value_t); // fallback estimate

#else
        throw std::logic_error("SparseLU: CuDSS is required. Rebuild with -DCUDDH_USE_CUDSS=ON.");
#endif
    }

    template <typename scalar_t, bool Complex>
    SparseLU<scalar_t, Complex>::~SparseLU() = default;

    template <typename scalar_t, bool Complex>
    SparseLU<scalar_t, Complex>::SparseLU(SparseLU &&) noexcept = default;

    template <typename scalar_t, bool Complex>
    SparseLU<scalar_t, Complex> &SparseLU<scalar_t, Complex>::operator=(SparseLU &&) noexcept = default;

    // ─── stats / print ───────────────────────────────────────────────────────────

    template <typename scalar_t, bool Complex>
    const SparseLUStats<scalar_t, Complex> &SparseLU<scalar_t, Complex>::stats() const
    {
        return _pimpl->stats;
    }

    template <typename scalar_t, bool Complex>
    void SparseLU<scalar_t, Complex>::print(std::ostream &os) const
    {
        const auto &s = _pimpl->stats;
        os << "SparseLU(n=" << s.n_rows << ", m=" << s.n_cols << ", nnz=" << s.nnz << ")\n"
           << "  analysis: " << s.analysis_seconds << " s\n"
           << "  factor:   " << s.factor_seconds << " s\n"
           << "  solves:   " << s.solve_calls << " calls, " << s.total_solve_seconds << " s total\n"
           << "  memory:   finalized=" << s.finalized_bytes << " B, factor=" << s.factor_bytes << " B\n";
    }

    // ─── solve ───────────────────────────────────────────────────────────────────

    template <typename scalar_t, bool Complex>
    bool SparseLU<scalar_t, Complex>::solve(const scalar_t *rhs, scalar_t *x)
    {
#ifdef CUDDH_HAS_CUDSS
        auto &p = *_pimpl;
        const int n = p.n_rows;

        const auto t0 = std::chrono::high_resolution_clock::now();

        if constexpr (!Complex)
        {
            // rhs and x are device pointers to n scalars.
            cudaMemcpy(thrust::raw_pointer_cast(p.b_dev.data()), rhs, static_cast<size_t>(n) * sizeof(value_t),
                       cudaMemcpyDeviceToDevice);

            cudssStatus_t st =
                cudssExecute(p.handle, CUDSS_PHASE_SOLVE, p.config, p.data, p.A_matrix, p.x_matrix, p.b_matrix);
            cudaDeviceSynchronize();

            cudaMemcpy(x, thrust::raw_pointer_cast(p.x_dev.data()), static_cast<size_t>(n) * sizeof(value_t),
                       cudaMemcpyDeviceToDevice);

            const auto t1 = std::chrono::high_resolution_clock::now();
            p.stats.total_solve_seconds += std::chrono::duration<double>(t1 - t0).count();
            ++p.stats.solve_calls;

            return st == CUDSS_STATUS_SUCCESS;
        }
        else
        {
            // rhs and x are device pointers in blocked format [re_0..re_{n-1}, im_0..im_{n-1}].
            // b_dev stores interleaved complex<scalar_t>; use thrust::complex for device kernels.
            using tcx = thrust::complex<scalar_t>;
            tcx *b_ptr = reinterpret_cast<tcx *>(thrust::raw_pointer_cast(p.b_dev.data()));
            forall(n, [=] __device__(int i) { b_ptr[i] = tcx(rhs[i], rhs[i + n]); });

            cudssStatus_t st =
                cudssExecute(p.handle, CUDSS_PHASE_SOLVE, p.config, p.data, p.A_matrix, p.x_matrix, p.b_matrix);
            cudaDeviceSynchronize();

            const tcx *x_ptr = reinterpret_cast<const tcx *>(thrust::raw_pointer_cast(p.x_dev.data()));
            forall(n, [=] __device__(int i) {
                x[i] = x_ptr[i].real();
                x[i + n] = x_ptr[i].imag();
            });

            const auto t1 = std::chrono::high_resolution_clock::now();
            p.stats.total_solve_seconds += std::chrono::duration<double>(t1 - t0).count();
            ++p.stats.solve_calls;

            return st == CUDSS_STATUS_SUCCESS;
        }
#else
        (void)rhs;
        (void)x;
        throw std::logic_error("SparseLU: CuDSS is required. Rebuild with -DCUDDH_USE_CUDSS=ON.");
#endif
    }

    // ─── Explicit instantiations ─────────────────────────────────────────────────

    template class SparseLU<float, false>;
    template class SparseLU<float, true>;
    template class SparseLU<double, false>;
    template class SparseLU<double, true>;

} // namespace cuddh
