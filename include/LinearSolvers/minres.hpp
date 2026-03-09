#pragma once
#include "LinearSolvers/SolverBase.hpp"
#include "cuddh_config.hpp"

namespace cuddh
{
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
    SolverResults minres(int n, double *x, const Operator<double> &A, const double *b, SolverParams opts = {});
    SolverResults minres(int n, float *x, const Operator<float> &A, const float *b, SolverParams opts = {});
} // namespace cuddh
