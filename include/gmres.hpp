#ifndef CUDDH_GMRES_HPP
#define CUDDH_GMRES_HPP

#include <iostream>
#include <iomanip>

#include "Tensor.hpp"
#include "linalg.hpp"
#include "Operator.hpp"

namespace cuddh
{
    struct solver_out
    {
        bool success;
        int num_iter;
        int num_matvec;
        int num_precond;
        std::vector<double> res_norm;
    };

    /// @brief GMRES(m) for solving A * x == b
    /// @param[in] n dimension of x
    /// @param[in,out] x length n. On entry, an initial estimate of the solution (or zero). On exit, the approximate solution x <- A \ b.
    /// @param[in] A an operator such that A.action(x, y) computes y <- A * x.
    /// @param[in] b length n. The right hand side of A * x == b.
    /// @param[in] Precond an operator such that Precond.action(x, y) computes y <- P * x where P ~ inv(A).
    /// @param[in] m the size of the Krylov space used at each iteration.
    /// @param[in] maxit the maximum number of iteration to convergence.
    /// @param[in] tol the relative tolerance for an acceptable solution. gmres stops when |A*x-b|/|b| < tol.
    /// @param[in] verbose if verbose == 1, gmres will print progress bar to cout; if verbose >= 2, gmres will print each iteration to cout; if verbose == 0, gmres is silent.
    solver_out gmres(int n, double * x, const Operator * A, const double * b, const Operator * Precond, int m, int maxit, double tol=1e-6, int verbose=0);
} // namespace cuddh


#endif