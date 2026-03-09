#include "LinearSolvers/gmres.hpp"

using namespace cuddh;

template <typename real_t>
GMRES<real_t>::GMRES(int n, const Operator<real_t> &A, const Operator<real_t> *M, int kdim, bool flexible)
    : BaseArnoldiSolver<real_t>(n, A, M, kdim, flexible), _r(n)
{}

template <typename real_t>
SolverResults GMRES<real_t>::solve(real_t *x, const real_t *b, SolverParams opts) const
{
    validate_params(opts);

    SolverLogger logger(opts.verbose, opts.maxit);

    const real_t bnrm = dla::norm(n, b);
    const real_t tol = std::max(opts.rtol * bnrm, opts.atol);

    auto r = thrust::raw_pointer_cast(_r.data());

    this->evaluate_residual(r, x, b);
    logger.log_matvec();

    real_t rnrm = dla::norm(n, r);
    logger.log_iteration(rnrm / bnrm);

    while (logger.num_iterations() < opts.maxit && rnrm > tol)
    {
        int m = std::min(kdim, opts.maxit - logger.num_iterations());
        this->arnoldi_cycle(logger, m, 0, x, r, rnrm, bnrm, tol);

        if (not flexible)
        {
            this->evaluate_residual(r, x, b);
            logger.log_matvec();
            rnrm = dla::norm(n, r);
        }
    }

    return logger.log_summary(rnrm / bnrm, rnrm <= tol);
}

template class GMRES<float>;
template class GMRES<double>;
