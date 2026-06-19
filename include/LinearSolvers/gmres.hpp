#pragma once

#include <thrust/device_vector.h>

#include "LinearSolvers/Arnoldi.hpp"
#include "LinearSolvers/SolverBase.hpp"
#include "Operator.hpp"

namespace cuddh
{
    template <typename real_t>
    class GMRES : public Solver<real_t>, private BaseArnoldiSolver<real_t>
    {
    public:
        GMRES(const Operator<real_t> &A, const Operator<real_t> *M = nullptr, int kdim = 20, bool flexible = false);

        SolverResults solve(real_t *x, const real_t *b, SolverParams opts = {}) const override;

    private:
        using base = BaseArnoldiSolver<real_t>;
        using base::flexible;
        using base::kdim;
        mutable thrust::device_vector<real_t> _r;
    };

    extern template class GMRES<float>;
    extern template class GMRES<double>;

    /// @brief GMRES(m) for solving A * x == b
    /// @param[in,out] x DEVICE. length n. On entry, an initial estimate of the solution (or zero). On exit, the
    /// approximate solution x <- A \ b.
    /// @param[in] A DEVICE KERNEL. an operator such that A.action(x, y) computes y <- A * x.
    /// @param[in] b DEVICE. length n. The right hand side of A * x == b.
    /// @param[in] Precond DEVICE KERNEL. an operator such that Precond.action(x, y) computes y <- P * x where P ~
    /// inv(A).
    inline SolverResults gmres(double *x, const Operator<double> &A, const double *b, int m = 20,
                               const Operator<double> *Precond = nullptr, SolverParams opts = {})
    {
        return GMRES<double>(A, Precond, m).solve(x, b, opts);
    }

    inline SolverResults gmres(float *x, const Operator<float> &A, const float *b, int m = 20,
                               const Operator<float> *Precond = nullptr, SolverParams opts = {})
    {
        return GMRES<float>(A, Precond, m).solve(x, b, opts);
    }

    /**
     * @brief Flexible GMRES(m) for solving A * x == b where the preconditioner can change at each iteration.
     *
     * @param x DEVICE. length n. On entry, an initial estimate of the solution (or zero). On exit, the approximate
     * solution x <- A \ b.
     * @param A DEVICE KERNEL. an operator such that A.action(x, y) computes y <- A * x.
     * @param b DEVICE. length n. The right hand side of A * x == b.
     * @param Precond Right preconditioner.
     * @return SolverResults
     */
    inline SolverResults fgmres(double *x, const Operator<double> &A, const double *b, int m = 20,
                                const Operator<double> *Precond = nullptr, SolverParams opts = {})
    {
        return GMRES<double>(A, Precond, m, true).solve(x, b, opts);
    }

    inline SolverResults fgmres(float *x, const Operator<float> &A, const float *b, int m = 20,
                                const Operator<float> *Precond = nullptr, SolverParams opts = {})
    {
        return GMRES<float>(A, Precond, m, true).solve(x, b, opts);
    }
} // namespace cuddh
