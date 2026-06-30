#pragma once

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

        void add_entry(int row, int col, scalar_t value)
        {
            if constexpr (Complex)
                add_value(row, col, value_t(value, scalar_t(0)));
            else
                add_value(row, col, value);
        }

        void add_entry(int row, int col, std::complex<scalar_t> value)
            requires(Complex)
        {
            add_value(row, col, value);
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

        void action(scalar_t c, const scalar_t *x, scalar_t *y) const override
        {
            require_state(SparseMatrixState::Finalized, "action");

            if constexpr (Complex)
                action_csr(value_t(c, scalar_t(0)), x, y);
            else
                action_csr(c, x, y);
        }

        void action(std::complex<scalar_t> c, const scalar_t *x, scalar_t *y) const
            requires(Complex)
        {
            require_state(SparseMatrixState::Finalized, "action");
            action_csr(c, x, y);
        }

        void action(const scalar_t *x, scalar_t *y) const override
        {
            const int n = this->ndof();
            std::fill(y, y + n, scalar_t{});
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
            require_state(SparseMatrixState::COOAssembly, "add_entry");

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

        void action_csr(value_t c, const scalar_t *x, scalar_t *y) const
        {
            ensure_finalized_storage();

            const int *row_ptr = _csr_row_ptr.host_read();
            const int *cols = _csr_cols.host_read();
            const value_t *vals = _csr_vals.host_read();

            if constexpr (!Complex)
            {
                for (int r = 0; r < _rows; ++r)
                    for (int k = row_ptr[r]; k < row_ptr[r + 1]; ++k)
                        y[r] += c * vals[k] * x[cols[k]];
            }
            else
            {
                std::vector<value_t> x_complex(static_cast<size_t>(_cols), value_t{});
                std::vector<value_t> y_complex(static_cast<size_t>(_rows), value_t{});

                for (int i = 0; i < _cols; ++i)
                    x_complex[i] = value_t(x[i], x[i + _cols]);

                for (int i = 0; i < _rows; ++i)
                    y_complex[i] = value_t(y[i], y[i + _rows]);

                for (int r = 0; r < _rows; ++r)
                {
                    value_t out = y_complex[r];
                    for (int k = row_ptr[r]; k < row_ptr[r + 1]; ++k)
                        out += c * vals[k] * x_complex[cols[k]];
                    y_complex[r] = out;
                }

                for (int i = 0; i < _rows; ++i)
                {
                    y[i] = y_complex[i].real();
                    y[i + _rows] = y_complex[i].imag();
                }
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
        BlockSparseMatrix() = default;

        explicit BlockSparseMatrix(std::vector<SparseMatrix<scalar_t, Complex>> blocks) : _blocks(std::move(blocks)) {}

        int size() const { return static_cast<int>(_blocks.size()); }
        SparseMatrixState state() const { return _state; }

        SparseMatrix<scalar_t, Complex> &block(int i) { return _blocks.at(i); }
        const SparseMatrix<scalar_t, Complex> &block(int i) const { return _blocks.at(i); }

        void add_entry(int block, int row, int col)
        {
            _blocks.at(block).add_entry(row, col);
            _state = SparseMatrixState::PatternAssembly;
        }

        void add_entry(int block, int row, int col, scalar_t value)
        {
            _blocks.at(block).add_entry(row, col, value);
            _state = SparseMatrixState::COOAssembly;
        }

        template <bool C = Complex>
            requires(C)
        void add_entry(int block, int row, int col, std::complex<scalar_t> value)
        {
            _blocks.at(block).add_entry(row, col, value);
            _state = SparseMatrixState::COOAssembly;
        }

        void finalize_pattern()
        {
            for (auto &b : _blocks)
                b.finalize_pattern();
            _state = SparseMatrixState::COOAssembly;
        }

        void finalize_values()
        {
            for (auto &b : _blocks)
                b.finalize_values();
            _state = SparseMatrixState::Finalized;
        }

    private:
        std::vector<SparseMatrix<scalar_t, Complex>> _blocks;
        SparseMatrixState _state{SparseMatrixState::PatternAssembly};
    };
} // namespace cuddh
