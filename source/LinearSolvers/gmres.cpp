#include "LinearSolvers/gmres.hpp"

using namespace cuddh;

class PreconditionedSystem : public Operator
{
public:
    PreconditionedSystem(int n, const Operator *A_, const Operator *P_) : q(n), A{A_}, P{P_} {}

    void action(double c, const double *x, double *y) const override
    {
        cuddh_verify(false, printf("Not implemented"));
    }

    void action(const double *x, double *y) const override
    {
        double *d_q = thrust::raw_pointer_cast(q.data());
        A->action(x, d_q);
        P->action(d_q, y);
    }

private:
    mutable thrust::universal_vector<double> q;
    const Operator *A;
    const Operator *P;
};

static void validate_opts(gmresParams &opts, bool ignore_m = false)
{
    cuddh_verify(ignore_m || opts.m > 0, printf("solver error: m must be positive\n"));
    cuddh_verify(opts.maxit > 0, printf("solver error: maxit must be positive\n"));
    cuddh_verify(opts.tol >= 0, printf("solver error: tol must be non-negative\n"));

    if (opts.m > opts.maxit)
        opts.m = opts.maxit;
}

template <typename scalar, typename OpType>
inline SolverResults t_gmres(int n, scalar *x, const OpType *A, const scalar *b, gmresParams &opts)
{
    constexpr scalar one = 1, zero = 0;

    validate_opts(opts);
    SolverLogger logger(opts.verbose, opts.maxit);

    const scalar bnrm = dla::norm(n, b);
    const scalar tol = std::max(opts.tol * bnrm, opts.atol);
    const int m1 = opts.m + 1;

    // DEVICE DATA:
    thrust::device_vector<scalar> _r(n, scalar{});
    thrust::device_vector<scalar> _V(n * m1, scalar{});
    scalar *r = thrust::raw_pointer_cast(_r.data());
    scalar *V = thrust::raw_pointer_cast(_V.data());

    // HOST DATA
    Matrix<scalar> H(m1, opts.m);
    Vec<scalar> sn(opts.m);
    Vec<scalar> cs(opts.m);
    Vec<scalar> eta(m1);

    A->action(x, r); // r <- A * x
    logger.log_matvec();
    dla::axpby(n, one, b, -one, r); // r <- b - r = b - A * x

    scalar rnrm = dla::norm(n, r);
    logger.log_iteration(rnrm / bnrm);

    while (logger.num_iterations() <= opts.maxit && rnrm > tol)
    {
        scalar *vk = V;
        scalar *vk1;

        dla::axpby(n, one / rnrm, r, zero, vk); // v[0] <- r / ||r||

        std::fill(eta.begin(), eta.end(), 0.0);
        eta(0) = rnrm;

        int k1 = 0;
        for (int k = 0; k < opts.m && logger.num_iterations() <= opts.maxit; ++k)
        {
            k1 = k + 1;
            vk = V + k * n;
            vk1 = vk + n;

            A->action(vk, vk1); // v[k+1] <- A * v[k]
            logger.log_matvec();

            for (int j = 0; j < k1; ++j)
            {
                const scalar *vj = V + j * n;
                H(j, k) = dla::dot(n, vk1, vj);
                dla::axpby(n, -H(j, k), vj, one, vk1); // v[k+1] <- v[k+1] - H(j, k) * v[j]
            }

            H(k1, k) = dla::norm(n, vk1);

            if (H(k1, k) == 0.0)
                break;

            dla::scal(n, one / H(k1, k), vk1); // v[k+1] <- v[k+1] / ||v[k+1]||

            for (int i = 0; i < k; ++i)
                apply_givens(H(i, k), H(i + 1, k), cs(i), sn(i));
            std::tie(cs(k), sn(k)) = compute_givens(H(k, k), H(k + 1, k));
            apply_givens(H(k, k), H(k + 1, k), cs(k), sn(k));
            apply_givens(eta(k), eta(k + 1), cs(k), sn(k));

            rnrm = std::abs(eta(k1));
            logger.log_iteration(rnrm / bnrm);

            if (rnrm <= tol)
                break;
        }

        solve_triu(k1, H.data(), H.shape(0), eta.data());
        for (int k = 0; k < k1; ++k)
            dla::axpby(n, eta(k), V + k * n, one, x); // x <- x + eta[k] * v[k]

        A->action(x, r); // r <- A * x
        logger.log_matvec();
        dla::axpby(n, one, b, -one, r); // r <- b - r = b - A * x

        rnrm = dla::norm(n, r);
    }

    return logger.log_summary(rnrm / bnrm, rnrm <= tol);
}

SolverResults cuddh::gmres(int n, double *x, const Operator *A, const double *b, gmresParams opts)
{
    return t_gmres<double>(n, x, A, b, opts);
}

SolverResults cuddh::gmres(int n, double *x, const Operator *A, const double *b, const Operator *P, gmresParams opts)
{
    PreconditionedSystem PA(n, A, P);

    host_device_dvec r0(n);
    double *d_r0 = r0.device_write();
    P->action(b, d_r0);

    return t_gmres<double>(n, x, &PA, d_r0, opts);
}

SolverResults cuddh::gmres(int n, float *x, const SinglePrecisionOperator *A, const float *b, gmresParams opts)
{
    return t_gmres<float>(n, x, A, b, opts);
}

cuddh::SolverResults cuddh::minres(int n, double *x, const Operator *A, const double *b, gmresParams opts)
{
    validate_opts(opts, true);
    SolverLogger logger(opts.verbose, opts.maxit);

    thrust::device_vector<double> _r(n), _v(n), _w(n, 0.0), _wp(n, 0.0), _vp(n, 0.0), _wpp(n, 0.0);
    double *r = thrust::raw_pointer_cast(_r.data());
    double *v = thrust::raw_pointer_cast(_v.data());
    double *w = thrust::raw_pointer_cast(_w.data());
    double *wp = thrust::raw_pointer_cast(_wp.data());
    double *vp = thrust::raw_pointer_cast(_vp.data());
    double *wpp = thrust::raw_pointer_cast(_wpp.data());

    const double bnrm = dla::norm(n, b);
    const double tol = std::max(opts.tol * bnrm, opts.atol);

    // r = b - A * x
    A->action(x, r);
    logger.log_matvec();
    dla::axpby(n, 1.0, b, -1.0, r);

    double phi = dla::norm(n, r);
    logger.log_iteration(phi / bnrm);

    // v = r / phi
    dla::axpby(n, 1.0 / phi, r, 0.0, v);

    double cp = 1.0, sp = 0.0;
    double c = 1.0, s = 0.0;
    double beta = 0.0;

    while (logger.num_iterations() <= opts.maxit)
    {
        // Lanczos step
        A->action(v, r); // r = A * v
        logger.log_matvec();
        double alpha = dla::dot(n, v, r); // (v, A*v)

        // r = A * v - alpha * v - beta * vp
        forall(n, [=] __device__(int i) { r[i] -= alpha * v[i] + beta * vp[i]; });

        // Givens
        double rho2 = sp * beta;
        double gamma = cp * beta;

        double rho1 = c * gamma + s * alpha;
        double delta = -s * gamma + c * alpha;

        beta = dla::norm(n, r);

        double rho3 = std::hypot(delta, beta);
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

        logger.log_iteration(std::abs(phi) / bnrm);

        if (std::abs(phi) <= tol)
            break;

        // prepare for next iteration
        forall(n, [=] __device__(int i) {
            wpp[i] = wp[i];
            wp[i] = w[i];
            vp[i] = v[i];
            v[i] = r[i] / beta;
        });
    }

    return logger.log_summary(std::abs(phi) / bnrm, std::abs(phi) <= tol);
}

SolverResults cuddh::fgmres(int n, double *x, const Operator *A, const double *b, const Operator *Precond,
                            gmresParams opts)
{
    validate_opts(opts);
    SolverLogger logger(opts.verbose, opts.maxit);

    const double bnrm = dla::norm(n, b);
    const double tol = std::max(opts.tol * bnrm, opts.atol);

    // DEVICE DATA:
    thrust::device_vector<double> _r(n, 0.0);
    thrust::device_vector<double> _V(n * (opts.m + 1), 0.0);
    thrust::device_vector<double> _Z(n * opts.m, 0.0);
    double *r = thrust::raw_pointer_cast(_r.data());
    double *V = thrust::raw_pointer_cast(_V.data());
    double *Z = thrust::raw_pointer_cast(_Z.data());

    // HOST DATA:
    Matrix<double> H(opts.m + 1, opts.m);
    Vec<double> sn(opts.m);
    Vec<double> cs(opts.m);
    Vec<double> eta(opts.m + 1);

    // compute initial residual
    A->action(x, r); // r <- A * x
    logger.log_matvec();
    dla::axpby(n, 1.0, b, -1.0, r); // r <- b - A * x

    double rnrm = dla::norm(n, r);
    logger.log_iteration(rnrm / bnrm);

    while (logger.num_iterations() <= opts.maxit)
    {
        dla::axpby(n, 1.0 / rnrm, r, 0.0, V); // v[0] <- r / ||r||
        eta(0) = rnrm;

        // Arnoldi process with variable preconditioner
        int k1 = 0;
        for (int k = 0; k < opts.m && logger.num_iterations() <= opts.maxit; ++k)
        {
            k1 = k + 1;
            const double *vk = V + k * n;
            double *vk1 = V + k1 * n;
            double *zk = Z + k * n;

            Precond->action(vk, zk); // z[k] <- Precond * v[k]
            A->action(zk, vk1);      // v[k+1] <- A * z[k]
            logger.log_matvec();

            // Modified Gram-Schmidt
            for (int j = 0; j <= k; ++j)
            {
                const double *vj = V + j * n;
                H(j, k) = dla::dot(n, vk1, vj);
                dla::axpby(n, -H(j, k), vj, 1.0, vk1); // v[k+1] <- v[k+1] - H(j, k) * v[j]
            }

            H(k + 1, k) = dla::norm(n, vk1);

            if (H(k + 1, k) < 1e-14)
                break;

            dla::scal(n, 1.0 / H(k + 1, k), vk1); // v[k+1] <- v[k+1] / ||v[k+1||

            for (int i = 0; i < k; ++i)
                apply_givens(H(i, k), H(i + 1, k), cs(i), sn(i));
            std::tie(cs(k), sn(k)) = compute_givens(H(k, k), H(k + 1, k));
            apply_givens(H(k, k), H(k + 1, k), cs(k), sn(k));
            apply_givens(eta(k), eta(k + 1), cs(k), sn(k));

            rnrm = std::abs(eta(k + 1));
            logger.log_iteration(rnrm / bnrm);

            if (rnrm <= tol)
                break;
        }

        solve_triu(k1, H.data(), H.shape(0), eta.data());
        for (int k = 0; k < k1; ++k)
            dla::axpby(n, eta(k), Z + k * n, 1.0, x); // x <- x + eta[k] * z[k]

        A->action(x, r); // r <- A * x
        logger.log_matvec();
        dla::axpby(n, 1.0, b, -1.0, r); // r <- b - r = b - A * x

        rnrm = dla::norm(n, r);

        if (rnrm <= tol)
            break;
    }

    return logger.log_summary(rnrm / bnrm, rnrm <= tol);
}
