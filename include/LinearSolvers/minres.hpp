#pragma once

#include <thrust/device_vector.h>

#include "LinearSolvers/SolverBase.hpp"
#include "Operator.hpp"

namespace cuddh
{
    template <typename real_t>
    class MINRES : public Solver<real_t>
    {
    public:
        MINRES(const Operator<real_t> &A)
            : Solver<real_t>(A.ndof()),
              A{A},
              _r(A.ndof()),
              _v(A.ndof()),
              _w(A.ndof()),
              _wp(A.ndof()),
              _vp(A.ndof()),
              _wpp(A.ndof())
        {}

        SolverResults solve(real_t *x, const real_t *b, SolverParams opts = {}) const override;

    private:
        const Operator<real_t> &A;
        mutable thrust::device_vector<real_t> _r, _v, _w, _wp, _vp, _wpp;
    };

    extern template class MINRES<float>;
    extern template class MINRES<double>;

    /**
     * @brief MINRES for solving A * x == b where A is symmetric (not necessarily positive definite).
     *
     * @param n dimension of x
     * @param x DEVICE. length n. On entry, an initial estimate of the solution (or zero). On exit, the approximate
     * solution x <- A \ b.
     * @param A DEVICE KERNEL. an operator such that A.action(x, y) computes y <- A * x.
     * @param b DEVICE. length n. The right hand side of A * x == b.
     * @param opts
     * @return SolverResults
     */
    inline SolverResults minres(double *x, const Operator<double> &A, const double *b, SolverParams opts = {})
    {
        return MINRES<double>(A).solve(x, b, opts);
    }

    inline SolverResults minres(float *x, const Operator<float> &A, const float *b, SolverParams opts = {})
    {
        return MINRES<float>(A).solve(x, b, opts);
    }
} // namespace cuddh
