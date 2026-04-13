#include "LinearSolvers/minres.hpp"

#include "forall.hpp"
#include "linalg.hpp"

using namespace cuddh;

template <typename real_t>
SolverResults MINRES<real_t>::solve(real_t *x, const real_t *b, SolverParams opts) const
{
    validate_params(opts);

    SolverLogger logger(opts.verbose, opts.maxit);

    const int n = this->ndof();

    real_t *r = thrust::raw_pointer_cast(_r.data());
    real_t *v = thrust::raw_pointer_cast(_v.data());
    real_t *w = thrust::raw_pointer_cast(_w.data());
    real_t *wp = thrust::raw_pointer_cast(_wp.data());
    real_t *vp = thrust::raw_pointer_cast(_vp.data());
    real_t *wpp = thrust::raw_pointer_cast(_wpp.data());

    dla::zeros(n, w);
    dla::zeros(n, wp);
    dla::zeros(n, vp);
    dla::zeros(n, wpp);

    const real_t bnrm = dla::norm(n, b);
    const real_t tol = std::max(opts.rtol * bnrm, opts.atol);

    // r = b - A * x
    A.action(x, r);
    logger.log_matvec();
    dla::axpby(n, 1.0, b, -1.0, r);

    real_t phi = dla::norm(n, r);
    real_t rnrm = phi;
    logger.log_iteration(rnrm / bnrm);

    // v = r / rnrm
    dla::axpby(n, 1.0 / rnrm, r, 0.0, v);

    real_t cp = 1.0, sp = 0.0;
    real_t c = 1.0, s = 0.0;
    real_t beta = 0.0;

    while (logger.num_iterations() <= opts.maxit)
    {
        // Lanczos step
        A.action(v, r); // r = A * v
        logger.log_matvec();
        real_t alpha = dla::dot(n, v, r); // (v, A*v)

        // r = A * v - alpha * v - beta * vp
        forall(n, [=] __device__(int i) { r[i] -= alpha * v[i] + beta * vp[i]; });

        // Givens
        real_t rho2 = sp * beta;
        real_t gamma = cp * beta;

        real_t rho1 = c * gamma + s * alpha;
        real_t delta = -s * gamma + c * alpha;

        beta = dla::norm(n, r);

        real_t rho3 = std::hypot(delta, beta);
        cp = c;
        sp = s;
        c = delta / rho3;
        s = beta / rho3;

        // update w and x
        forall(n, [=] __device__(int i) {
            w[i] = (v[i] - rho1 * wp[i] - rho2 * wpp[i]) / rho3;
            x[i] += c * phi * w[i];
        });

        // update norm and check convergence
        phi = -s * phi;
        rnrm = std::abs(phi);

        logger.log_iteration(rnrm / bnrm);

        if (rnrm <= tol)
            break;

        // prepare for next iteration
        forall(n, [=] __device__(int i) {
            wpp[i] = wp[i];
            wp[i] = w[i];
            vp[i] = v[i];
            v[i] = r[i] / beta;
        });
    }

    return logger.log_summary(rnrm / bnrm, rnrm <= tol);
}

namespace cuddh
{
    template class MINRES<float>;
    template class MINRES<double>;
} // namespace cuddh
