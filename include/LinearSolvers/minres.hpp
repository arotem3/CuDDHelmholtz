#pragma once
#include "LinearSolvers/SolverBase.hpp"
#include "cuddh_config.hpp"

namespace cuddh
{
    template <typename real_t>
    class MINRES
    {
    public:
        MINRES(int n, const Operator<real_t> &A) : n{n}, A{A}, _r(n), _v(n), _w(n), _wp(n), _vp(n), _wpp(n) {}

        SolverResults solve(real_t *x, const real_t *b, SolverParams opts = {}) const;

    private:
        int n;
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
    inline SolverResults minres(int n, double *x, const Operator<double> &A, const double *b, SolverParams opts = {})
    {
        return MINRES<double>(n, A).solve(x, b, opts);
    }

    inline SolverResults minres(int n, float *x, const Operator<float> &A, const float *b, SolverParams opts = {})
    {
        return MINRES<float>(n, A).solve(x, b, opts);
    }
} // namespace cuddh
