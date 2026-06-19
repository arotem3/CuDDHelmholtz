#pragma once

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "LinearSolvers/KrylovHelpers.hpp"
#include "LinearSolvers/SolverBase.hpp"
#include "Operator.hpp"
#include "linalg.hpp"

namespace cuddh
{
    template <typename real_t>
    class BaseArnoldiSolver
    {
    public:
        BaseArnoldiSolver(const Operator<real_t> &A, const Operator<real_t> *M, int kdim, bool flexible);

        int arnoldi_cycle(SolverLogger &logger, int m, int k, real_t *x, real_t *r, real_t &rnrm, real_t bnrm,
                          real_t tol) const;

        inline void evaluate_residual(real_t *r, const real_t *x, const real_t *b) const
        {
            auto res = [&](const real_t *in, real_t *out) {
                A->action(in, out);
                dla::axpby(A->ndof(), real_t(1.0), b, real_t(-1.0), out);
            };

            if (M && !flexible)
            {
                real_t *tmp = thrust::raw_pointer_cast(_Z.data());
                res(x, tmp);
                M->action(tmp, r);
            }
            else
                res(x, r);
        }

    protected:
        const int kdim;
        const bool flexible;

        const Operator<real_t> *A;
        const Operator<real_t> *M;

        mutable thrust::device_vector<real_t> _Z; // flexible ? (n, kdim) : (n)
        mutable thrust::device_vector<real_t> _W; // (n, kdim+1)

        mutable Matrix<real_t> H;                // (kdim+1, kdim)
        mutable Matrix<real_t> Hqr;              // (kdim+1, kdim)
        mutable thrust::host_vector<real_t> eta; // kdim+1

        mutable Vec<real_t> cs; // (kdim)
        mutable Vec<real_t> sn; // (kdim)
    };

    extern template class BaseArnoldiSolver<float>;
    extern template class BaseArnoldiSolver<double>;
} // namespace cuddh
