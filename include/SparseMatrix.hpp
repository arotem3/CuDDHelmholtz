#pragma once

#include <thrust/complex.h>

#include <algorithm>
#include <chrono>
#include <complex>
#include <cstdint>
#include <memory>
#include <ostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "HostDeviceArray.hpp"
#include "Operator.hpp"
#include "forall.hpp"
#include "linalg.hpp"

namespace cuddh
{
    enum class SparseMatrixState
    {
        PatternAssembly,
        COOAssembly,
        Finalized,
    };

    template <typename scalar_t, bool Complex = false>
    class SparseLU;

    template <typename scalar_t, bool Complex = false>
    class SparseBlockLU;

    template <typename scalar_t, bool Complex = false>
    class SparseMatrix : public Operator<scalar_t>
    {
    public:
        using value_t = std::conditional_t<Complex, std::complex<scalar_t>, scalar_t>;

        explicit SparseMatrix(int n_rows, int n_cols, int reserve_nnz = 0)
            : Operator<scalar_t>(Complex ? 2 * n_rows : n_rows), _rows{n_rows}, _cols{n_cols}
        {
            if (n_rows < 0 || n_cols < 0)
                throw std::invalid_argument("SparseMatrix: n_rows and n_cols must be non-negative");

            if (reserve_nnz > 0)
                _pattern_entries.reserve(reserve_nnz);
        }

        int n_rows() const { return _rows; }
        int n_cols() const { return _cols; }
        int nnz() const { return _nnz; }
        SparseMatrixState state() const { return _state; }

        void add_entry(int row, int col)
        {
            check_rowcol(row, col);
            require_state(SparseMatrixState::PatternAssembly, "add_entry");
            _pattern_entries.push_back({row, col});
        }

        void set_value(int row, int col, scalar_t value)
        {
            if constexpr (Complex)
                add_value(row, col, value_t(value, scalar_t(0)));
            else
                add_value(row, col, value);
        }

        void set_value(int row, int col, std::complex<scalar_t> value)
            requires(Complex)
        {
            add_value(row, col, value);
        }

        void finalize_pattern()
        {
            require_state(SparseMatrixState::PatternAssembly, "finalize_pattern");

            std::sort(
                _pattern_entries.begin(), _pattern_entries.end(),
                [](const PatternEntry &a, const PatternEntry &b) { return less_entry(a.row, a.col, b.row, b.col); });

            auto unique_end = std::unique(
                _pattern_entries.begin(), _pattern_entries.end(),
                [](const PatternEntry &a, const PatternEntry &b) { return a.row == b.row && a.col == b.col; });
            _pattern_entries.erase(unique_end, _pattern_entries.end());

            _nnz = static_cast<int>(_pattern_entries.size());
            _coo_rows.resize(_nnz);
            _coo_cols.resize(_nnz);
            _coo_vals.resize(_nnz);

            int *rows = _coo_rows.host_write();
            int *cols = _coo_cols.host_write();
            value_t *vals = _coo_vals.host_write();
            for (int i = 0; i < _nnz; ++i)
            {
                rows[i] = _pattern_entries[i].row;
                cols[i] = _pattern_entries[i].col;
                vals[i] = value_t{};
            }

            _pattern_entries.clear();
            _pattern_entries.shrink_to_fit();
            _state = SparseMatrixState::COOAssembly;
        }

        void finalize_values()
        {
            if (_state == SparseMatrixState::Finalized)
                return;

            require_state(SparseMatrixState::COOAssembly, "finalize_values");

            build_csr_from_coo();

            // COO is always dropped once CSR is built.
            _coo_rows.resize(0);
            _coo_cols.resize(0);
            _coo_vals.resize(0);

            _state = SparseMatrixState::Finalized;
        }

        size_t finalized_storage_bytes() const
        {
            if (!_finalized_storage_present)
                return 0;

            return static_cast<size_t>(_rows + 1) * sizeof(int) + static_cast<size_t>(_nnz) * sizeof(int) +
                   static_cast<size_t>(_nnz) * sizeof(value_t);
        }

        // x and y are device pointers. y <- y + c * A * x.
        void action(scalar_t c, const scalar_t *x, scalar_t *y) const override
        {
            require_state(SparseMatrixState::Finalized, "action");
            if constexpr (Complex)
                action_csr(value_t(c, scalar_t(0)), x, y);
            else
                action_csr(c, x, y);
        }

        // x and y are device pointers. y <- y + c * A * x (complex scaling).
        void action(std::complex<scalar_t> c, const scalar_t *x, scalar_t *y) const
            requires(Complex)
        {
            require_state(SparseMatrixState::Finalized, "action");
            action_csr(c, x, y);
        }

        // x and y are device pointers. y <- A * x.
        void action(const scalar_t *x, scalar_t *y) const override
        {
            require_state(SparseMatrixState::Finalized, "action");
            dla::zeros(this->ndof(), y);
            action(scalar_t{1}, x, y);
        }

        __host__ __device__ static std::uint64_t key(int row, int col)
        {
            return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(row)) << 32U) |
                   static_cast<std::uint32_t>(col);
        }

        __host__ __device__ static int find_slot(const int *rows, const int *cols, int nnz, int row, int col)
        {
            int lo = 0;
            int hi = nnz;
            while (lo < hi)
            {
                const int mid = lo + (hi - lo) / 2;
                if (less_entry(rows[mid], cols[mid], row, col))
                    lo = mid + 1;
                else
                    hi = mid;
            }

            if (lo < nnz && rows[lo] == row && cols[lo] == col)
                return lo;
            return -1;
        }

    private:
        struct PatternEntry
        {
            int row;
            int col;
        };

        static __host__ __device__ bool less_entry(int row_a, int col_a, int row_b, int col_b)
        {
            return row_a < row_b || (row_a == row_b && col_a < col_b);
        }

        void add_value(int row, int col, value_t value)
        {
            check_rowcol(row, col);
            require_state(SparseMatrixState::COOAssembly, "set_value");

            const int slot = find_slot(_coo_rows.host_read(), _coo_cols.host_read(), _nnz, row, col);
            if (slot < 0)
                throw std::logic_error("SparseMatrix: entry is not part of the finalized pattern");

            value_t *vals = _coo_vals.host_read_write();
            vals[slot] += value;
        }

        void check_rowcol(int row, int col) const
        {
            if (row < 0 || row >= _rows || col < 0 || col >= _cols)
                throw std::out_of_range("SparseMatrix: row/col index out of range");
        }

        void require_state(SparseMatrixState expected, const char *op) const
        {
            if (_state != expected)
                throw std::logic_error(std::string(op) + ": invalid state");
        }

        void ensure_finalized_storage() const
        {
            if (!_finalized_storage_present)
                throw std::logic_error("SparseMatrix: finalized CSR storage is not available");
        }

        void build_csr_from_coo()
        {
            const int *rows = _coo_rows.host_read();
            const int *cols = _coo_cols.host_read();
            const value_t *vals = _coo_vals.host_read();

            // Count non-zeros, dropping structural zeros so that sparse
            // operators like MassMatrix (diagonal) can assemble into a full
            // FEM-space pattern without wasting storage.
            int csr_nnz = 0;
            for (int i = 0; i < _nnz; ++i)
                if (vals[i] != value_t{})
                    ++csr_nnz;

            _csr_row_ptr.resize(_rows + 1);
            _csr_cols.resize(csr_nnz);
            _csr_vals.resize(csr_nnz);

            int *row_ptr = _csr_row_ptr.host_write();
            int *csr_cols = _csr_cols.host_write();
            value_t *csr_vals = _csr_vals.host_write();

            std::fill(row_ptr, row_ptr + _rows + 1, 0);
            for (int i = 0; i < _nnz; ++i)
                if (vals[i] != value_t{})
                    row_ptr[rows[i] + 1]++;

            for (int r = 0; r < _rows; ++r)
                row_ptr[r + 1] += row_ptr[r];

            int k = 0;
            for (int i = 0; i < _nnz; ++i)
            {
                if (vals[i] != value_t{})
                {
                    csr_cols[k] = cols[i];
                    csr_vals[k] = vals[i];
                    ++k;
                }
            }

            _nnz = csr_nnz;
            _finalized_storage_present = true;
        }

    public:
        // x and y are device pointers. Computes y += c * A * x on GPU.
        void action_csr(value_t c, const scalar_t *x, scalar_t *y) const
        {
            ensure_finalized_storage();

            const int *rp = _csr_row_ptr.device_read();
            const int *ci = _csr_cols.device_read();

            if constexpr (!Complex)
            {
                const scalar_t *cv = _csr_vals.device_read();
                const int nr = _rows;
                forall(nr, [=] __device__(int r) {
                    scalar_t acc{};
                    for (int k = rp[r]; k < rp[r + 1]; ++k)
                        acc += cv[k] * x[ci[k]];
                    y[r] += c * acc;
                });
            }
            else
            {
                // x and y are in blocked format: [re_0..re_{n-1}, im_0..im_{n-1}].
                using tcx = thrust::complex<scalar_t>;
                const tcx *cv = reinterpret_cast<const tcx *>(_csr_vals.device_read());
                const tcx c_cx(c.real(), c.imag());
                const int nr = _rows;
                const int nc = _cols;
                forall(nr, [=] __device__(int r) {
                    tcx acc{};
                    for (int k = rp[r]; k < rp[r + 1]; ++k)
                    {
                        const int col = ci[k];
                        acc += cv[k] * tcx(x[col], x[col + nc]);
                    }
                    const tcx res = c_cx * acc;
                    y[r] += res.real();
                    y[r + nr] += res.imag();
                });
            }
        }

    private:
        int _rows;
        int _cols;
        int _nnz{0};
        bool _finalized_storage_present{false};
        SparseMatrixState _state{SparseMatrixState::PatternAssembly};

        std::vector<PatternEntry> _pattern_entries;
        HostDeviceArray<int> _coo_rows;
        HostDeviceArray<int> _coo_cols;
        HostDeviceArray<value_t> _coo_vals;

        HostDeviceArray<int> _csr_row_ptr;
        HostDeviceArray<int> _csr_cols;
        HostDeviceArray<value_t> _csr_vals;

        friend class SparseLU<scalar_t, Complex>;
    };

    template <typename scalar_t, bool Complex = false>
    struct SparseLUStats
    {
        int n_rows{0};
        int n_cols{0};
        int nnz{0};
        size_t finalized_bytes{0};
        size_t factor_bytes{0};
        double analysis_seconds{0.0};
        double factor_seconds{0.0};
        double total_solve_seconds{0.0};
        int solve_calls{0};
    };

    template <typename scalar_t, bool Complex>
    class SparseLU
    {
    public:
        using value_t = typename SparseMatrix<scalar_t, Complex>::value_t;

        explicit SparseLU(const SparseMatrix<scalar_t, Complex> &A);

        ~SparseLU();
        SparseLU(SparseLU &&) noexcept;
        SparseLU &operator=(SparseLU &&) noexcept;
        SparseLU(const SparseLU &) = delete;
        SparseLU &operator=(const SparseLU &) = delete;

        const SparseLUStats<scalar_t, Complex> &stats() const;

        bool solve(const scalar_t *rhs, scalar_t *x);

        void print(std::ostream &os) const;

    private:
        struct Impl;
        std::unique_ptr<Impl> _pimpl;
    };

    template <typename scalar_t, bool Complex>
    inline std::ostream &operator<<(std::ostream &os, const SparseLU<scalar_t, Complex> &lu)
    {
        lu.print(os);
        return os;
    }

    template <typename scalar_t, bool Complex = false>
    class BlockSparseMatrix
    {
    public:
        using value_t = std::conditional_t<Complex, std::complex<scalar_t>, scalar_t>;

        BlockSparseMatrix() = default;

        BlockSparseMatrix(int n_blocks, const int *block_sizes)
            : _n_blocks(n_blocks),
              _block_sizes(block_sizes, block_sizes + n_blocks),
              _coo_pattern(n_blocks),
              _rp_offsets(n_blocks + 1, 0),
              _nz_offsets(n_blocks + 1, 0)
        {
            if (n_blocks < 0)
                throw std::invalid_argument("BlockSparseMatrix: n_blocks must be non-negative");
        }

        int n_blocks() const { return _n_blocks; }
        int block_size(int b) const { return _block_sizes[b]; }
        int block_nnz(int b) const { return _nz_offsets[b + 1] - _nz_offsets[b]; }
        int total_nnz() const { return _nz_offsets[_n_blocks]; }
        SparseMatrixState state() const { return _state; }

        // Flat CSR data — valid after finalize_pattern().
        // Block b's row ptrs occupy row_ptrs()[rp_offset(b) .. rp_offset(b+1)-1] (local offsets starting from 0).
        // Block b's non-zeros occupy col_indices()/values()[nz_offset(b) .. nz_offset(b+1)-1].
        const int *row_ptrs() const { return _row_ptrs.data(); }
        const int *col_indices() const { return _col_idx.data(); }
        const value_t *values() const { return _values.data(); }
        int rp_offset(int b) const { return _rp_offsets[b]; }
        int nz_offset(int b) const { return _nz_offsets[b]; }

        void add_entry(int b, int row, int col)
        {
            if (_state != SparseMatrixState::PatternAssembly)
                throw std::logic_error("BlockSparseMatrix::add_entry: invalid state");
            _coo_pattern[b].emplace_back(row, col);
        }

        void set_value(int b, int row, int col, value_t val)
        {
            if (_state != SparseMatrixState::COOAssembly)
                throw std::logic_error("BlockSparseMatrix::set_value: invalid state");
            const int rp_base = _rp_offsets[b];
            const int nz_base = _nz_offsets[b];
            const int loc0 = _row_ptrs[rp_base + row];
            const int loc1 = _row_ptrs[rp_base + row + 1];
            const int *ci = _col_idx.data() + nz_base + loc0;
            const int len = loc1 - loc0;
            const int pos = static_cast<int>(std::lower_bound(ci, ci + len, col) - ci);
            if (pos >= len || ci[pos] != col)
                throw std::invalid_argument("BlockSparseMatrix::set_value: (row, col) not in pattern");
            _values[nz_base + loc0 + pos] += val;
        }

        void finalize_pattern()
        {
            _rp_offsets[0] = 0;
            _nz_offsets[0] = 0;
            for (int b = 0; b < _n_blocks; ++b)
            {
                auto &coo = _coo_pattern[b];
                std::sort(coo.begin(), coo.end());
                coo.erase(std::unique(coo.begin(), coo.end()), coo.end());
                _rp_offsets[b + 1] = _rp_offsets[b] + _block_sizes[b] + 1;
                _nz_offsets[b + 1] = _nz_offsets[b] + static_cast<int>(coo.size());
            }

            _row_ptrs.resize(_rp_offsets[_n_blocks], 0);
            _col_idx.resize(_nz_offsets[_n_blocks]);
            _values.assign(_nz_offsets[_n_blocks], value_t{});

            for (int b = 0; b < _n_blocks; ++b)
            {
                const int n = _block_sizes[b];
                const auto &coo = _coo_pattern[b];
                int *rp = _row_ptrs.data() + _rp_offsets[b];
                int *ci = _col_idx.data() + _nz_offsets[b];

                rp[0] = 0;
                int cur_row = 0, k = 0;
                for (const auto &[row, col] : coo)
                {
                    while (cur_row < row)
                        rp[++cur_row] = k;
                    ci[k++] = col;
                }
                while (cur_row < n)
                    rp[++cur_row] = static_cast<int>(coo.size());
            }

            _coo_pattern.clear();
            _coo_pattern.shrink_to_fit();
            _state = SparseMatrixState::COOAssembly;
        }

        void finalize_values() { _state = SparseMatrixState::Finalized; }

    private:
        int _n_blocks{0};
        std::vector<int> _block_sizes;
        std::vector<int> _rp_offsets;
        std::vector<int> _nz_offsets;

        std::vector<std::vector<std::pair<int, int>>> _coo_pattern; // cleared by finalize_pattern()

        std::vector<int> _row_ptrs;
        std::vector<int> _col_idx;
        std::vector<value_t> _values;

        SparseMatrixState _state{SparseMatrixState::PatternAssembly};
    };

    template <typename scalar_t, bool Complex = false>
    struct SparseBlockLUStats
    {
        int n_blocks{0};
        int max_n{0};
        int total_nnz{0};
        size_t factor_bytes{0};
        double analysis_seconds{0.0};
        double factor_seconds{0.0};
        double total_solve_seconds{0.0};
        int solve_calls{0};
    };

    /// @brief CuDSS non-uniform batch LU factorization for a collection of sparse matrices.
    ///
    /// All blocks are analyzed and factored in a single CuDSS batch call.  Solves are also
    /// issued as a single batch call, making this the preferred interface for DD subdomain solves
    /// where each subdomain has its own sparse matrix of potentially different size.
    ///
    /// `solve(d_rhs, d_x)` accepts and returns contiguous device buffers in blocked-complex
    /// layout: block p occupies bytes at stride `2 * max_size` from the buffer start, with
    /// re_0..re_{n_p-1} followed by im_0..im_{n_p-1} (padded to max_size with zeros).
    ///
    /// Requires CuDSS. Fails to compile if the library is built without `-DCUDDH_USE_CUDSS=ON`.
    template <typename scalar_t, bool Complex>
    class SparseBlockLU
    {
    public:
        using value_t = typename SparseMatrix<scalar_t, Complex>::value_t;

        /// @brief Construct from a finalized BlockSparseMatrix; runs batch analyze + factor.
        explicit SparseBlockLU(const BlockSparseMatrix<scalar_t, Complex> &blocks);

        ~SparseBlockLU();
        SparseBlockLU(SparseBlockLU &&) noexcept;
        SparseBlockLU &operator=(SparseBlockLU &&) noexcept;
        SparseBlockLU(const SparseBlockLU &) = delete;
        SparseBlockLU &operator=(const SparseBlockLU &) = delete;

        const SparseBlockLUStats<scalar_t, Complex> &stats() const;

        /// @brief Solve A_p * x_p = rhs_p for all blocks p (one CuDSS batch call).
        /// @param d_rhs Device buffer: block p at offset p*2*max_n, layout [re; im] padded to max_n.
        /// @param d_x   Device output buffer, same layout as d_rhs.
        /// @return true on success; throws on CuDSS error.
        bool solve(const scalar_t *d_rhs, scalar_t *d_x);

        void print(std::ostream &os) const;

    private:
        struct Impl;
        std::unique_ptr<Impl> _pimpl;
    };

    template <typename scalar_t, bool Complex>
    inline std::ostream &operator<<(std::ostream &os, const SparseBlockLU<scalar_t, Complex> &lu)
    {
        lu.print(os);
        return os;
    }
} // namespace cuddh
