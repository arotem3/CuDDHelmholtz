#pragma once

#include <complex>

namespace cuddh
{
    template <typename scalar_t, bool Complex>
    class SparseMatrix;

    template <typename scalar_t>
    class Operator
    {
    public:
        Operator(int n) : _n{n} {}
        virtual ~Operator() = default;

        /// @brief y <- y + c * A * x
        virtual void action(scalar_t c, const scalar_t *x, scalar_t *y) const = 0;

        /// @brief y <- A * x
        virtual void action(const scalar_t *x, scalar_t *y) const = 0;

        /// @brief Optional explicit assembly hook: S <- S + c * A.
        virtual bool assemble(scalar_t, SparseMatrix<scalar_t, false> &) const { return false; }

        /// @brief Optional explicit assembly hook for blocked-complex direct solves.
        virtual bool assemble(std::complex<scalar_t>, SparseMatrix<scalar_t, true> &) const { return false; }

        constexpr int ndof() const { return _n; }

    protected:
        constexpr void set_size(int n) { _n = n; }

    private:
        int _n;
    };
} // namespace cuddh
