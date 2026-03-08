#ifndef CUDDH_GMRES_HPP
#define CUDDH_GMRES_HPP

#include "LinearSolvers/SolverBase.hpp"

namespace cuddh
{
    struct gmresParams
    {
        enum Verbosity
        {
            Silent,
            ProgressBar,
            Iteration
        };

        int m = 20;        // number of vectors in the Krylov space used at each iteration of GMRES
        int maxit = 100;   // maximum number of iterations of GMRES
        double tol = 1e-3; // relative tolerance for an acceptable solution. gmres stops when |A*x-b|/|b| < tol.
        double atol = 0.0; // absolute tolerance for an acceptable solution. gmres stops when |A*x-b| < atol.
        Verbosity verbose = Silent; // 0: silent, 1: progress bar, 2: one line per iteration
    };

    /// @brief GMRES(m) for solving A * x == b
    /// @param[in] n dimension of x
    /// @param[in,out] x DEVICE. length n. On entry, an initial estimate of the solution (or zero). On exit, the
    /// approximate solution x <- A \ b.
    /// @param[in] A DEVICE KERNEL. an operator such that A.action(x, y) computes y <- A * x.
    /// @param[in] b DEVICE. length n. The right hand side of A * x == b.
    /// @param[in] Precond DEVICE KERNEL. an operator such that Precond.action(x, y) computes y <- P * x where P ~
    /// inv(A).
    /// each iteration to cout; if verbose == 0, gmres is silent.
    SolverResults gmres(int n, double *x, const Operator *A, const double *b, const Operator *Precond,
                        gmresParams opts = {});
    SolverResults gmres(int n, double *x, const Operator *A, const double *b, gmresParams opts = {});

    SolverResults gmres(int n, float *x, const SinglePrecisionOperator *A, const float *b, gmresParams opts = {});

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
    SolverResults minres(int n, double *x, const Operator *A, const double *b, gmresParams opts = {});

    /**
     * @brief Flexible GMRES(m) for solving A * x == b where the preconditioner can change at each iteration.
     *
     * @param n dimension of x
     * @param x DEVICE. length n. On entry, an initial estimate of the solution (or zero). On exit, the approximate
     * solution x <- A \ b.
     * @param A DEVICE KERNEL. an operator such that A.action(x, y) computes y <- A * x.
     * @param b DEVICE. length n. The right hand side of A * x == b.
     * @param Precond Right preconditioner.
     * @return SolverResults
     */
    SolverResults fgmres(int n, double *x, const Operator *A, const double *b, const Operator *Precond,
                         gmresParams opts = {});
} // namespace cuddh

#endif