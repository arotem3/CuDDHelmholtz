#include "gmres.hpp"

using namespace cuddh;

// lapack routine for triangular solve
extern "C" void dtrsv_(char *uplo, char *trans, char *diag, int *n, const double *a, int *lda, double *x, int *inc_x);
extern "C" void strsv_(char *uplo, char *trans, char *diag, int *n, const float *a, int *lda, float *x, int *inc_x);

template <typename scalar>
static void givens_rotations(scalar *h, scalar *cs, scalar *sn, int k)
{
    for (int i = 0; i < k; ++i)
    {
        const scalar h1 = h[i], h2 = h[i + 1];
        h[i] = cs[i] * h1 + sn[i] * h2;
        h[i + 1] = -sn[i] * h1 + cs[i] * h2;
    }

    scalar t = std::hypot(h[k], h[k + 1]);
    cs[k] = h[k] / t;
    sn[k] = h[k + 1] / t;

    h[k] = cs[k] * h[k] + sn[k] * h[k + 1];
    h[k + 1] = 0.0;
}

// solve R * x == b inplace on b where R is upper triangular
static void solve_upper_triangular(int N, const double *R, int ldr, double *b)
{
    int i_one = 1.0;
    char uplo[] = "u";
    char trans[] = "n";
    char diag[] = "n";
    dtrsv_(uplo, trans, diag, &N, R, &ldr, b, &i_one);
}

static void solve_upper_triangular(int N, const float *R, int ldr, float *b)
{
    int i_one = 1.0;
    char uplo[] = "u";
    char trans[] = "n";
    char diag[] = "n";
    strsv_(uplo, trans, diag, &N, R, &ldr, b, &i_one);
}

class ProgressBar
{
public:
    ProgressBar(int nt_) : it{0}, nt{nt_}, progress(30, ' ') {}

    void operator++()
    {
        it = std::min(it + 1, nt - 1);
        progress.at(30 * it / nt) = '#';
    }

    const std::string &get() const
    {
        return progress;
    }

private:
    int it;
    const int nt;
    std::string progress;
};

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

static void validate_opts(SolverParams &opts)
{
    cuddh_verify(opts.m > 0, printf("solver error: m must be positive\n"));
    cuddh_verify(opts.maxit > 0, printf("solver error: maxit must be positive\n"));
    cuddh_verify(opts.tol >= 0, printf("solver error: tol must be non-negative\n"));

    if (opts.m > opts.maxit)
        opts.m = opts.maxit;
}

static std::string format_time(double t)
{
    if (t < 1e-3)
        return std::format("{:.2f}µs", 1e6 * t);
    else if (t < 1.0)
        return std::format("{:.2f}ms", 1e3 * t);
    else if (t < 60.0)
        return std::format("{:.2f}s", t);
    else if (t < 3600.0)
    {
        int minutes = static_cast<int>(t) / 60;
        int seconds = static_cast<int>(t) % 60;
        return std::format("{:02d}m {:02d}s", minutes, seconds);
    }
    else
    {
        int hours = static_cast<int>(t) / 3600;
        int minutes = (static_cast<int>(t) % 3600) / 60;
        int seconds = static_cast<int>(t) % 60;
        return std::format("{:02d}h {:02d}m {:02d}s", hours, minutes, seconds);
    }
}

template <typename scalar, typename OpType>
inline SolverResults t_gmres(int n, scalar *x, const OpType *A, const scalar *b, SolverParams &opts)
{
    constexpr scalar one = 1, zero = 0;

    validate_opts(opts);

    const scalar bnrm = cuddh::norm(n, b);
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

    SolverResults out;
    out.res_norm.reserve((opts.maxit + 1) * opts.m);
    out.time.reserve((opts.maxit + 1) * opts.m);
    out.num_matvec = 0;
    out.success = false;

    A->action(x, r); // r <- A * x
    out.num_matvec++;
    axpby(n, one, b, -one, r); // r <- b - r = b - A * x

    scalar rnrm = cuddh::norm(n, r);

    out.res_norm.push_back((double)rnrm);
    out.time.push_back(0.0);
    auto t0 = std::chrono::high_resolution_clock::now();

    if (rnrm <= opts.tol * bnrm + opts.atol)
    {
        out.success = true;

        if (opts.verbose != SolverParams::Silent)
        {
            std::cout << "After 0 iterations, GMRES achieved rel. residual of " << out.res_norm.back() / bnrm
                      << std::endl;
            std::cout << "GMRES successfully converged within desired tolerance." << std::endl;
        }

        return out;
    }

    ProgressBar bar(opts.maxit);
    if (opts.verbose != SolverParams::Silent)
        std::cout << std::setprecision(5) << std::scientific;

    int it = 1;
    while (it <= opts.maxit)
    {
        scalar *vk = V;
        scalar *vk1;

        cuddh::axpby(n, one / rnrm, r, zero, vk); // v[0] <- r / ||r||

        std::fill(eta.begin(), eta.end(), 0.0);
        eta(0) = rnrm;

        int k1 = 0;
        for (int k = 0; k < opts.m && it <= opts.maxit; ++k, ++it)
        {
            k1 = k + 1;
            vk = V + k * n;
            vk1 = vk + n;

            A->action(vk, vk1); // v[k+1] <- A * v[k]
            out.num_matvec++;

            for (int j = 0; j < k1; ++j)
            {
                const scalar *vj = V + j * n;
                H(j, k) = cuddh::dot(n, vk1, vj);
                cuddh::axpby(n, -H(j, k), vj, one, vk1); // v[k+1] <- v[k+1] - H(j, k) * v[j]
            }

            H(k1, k) = cuddh::norm(n, vk1);

            if (H(k1, k) == 0.0)
                break;

            cuddh::scal(n, one / H(k1, k), vk1); // v[k+1] <- v[k+1] / ||v[k+1||

            givens_rotations(&H(0, k), (scalar *)cs, (scalar *)sn, k);
            eta(k1) = -sn(k) * eta(k);
            eta(k) = cs(k) * eta(k);

            rnrm = std::abs(eta(k1));
            out.res_norm.push_back((double)rnrm);

            if (opts.verbose == SolverParams::ProgressBar)
            {
                ++bar;
                std::cout << "[" << bar.get() << "] || iteration " << std::setw(10) << it << " / " << opts.maxit
                          << " || rel. res. = " << std::setw(10) << rnrm / bnrm << "\r" << std::flush;
            }
            else if (opts.verbose == SolverParams::Iteration)
            {
                std::cout << "iteration " << std::setw(10) << it << " / " << opts.maxit
                          << " || rel. res. = " << std::setw(10) << rnrm / bnrm << std::endl;
            }

            if (rnrm <= opts.tol * bnrm + opts.atol)
                break;

            auto t1 = std::chrono::high_resolution_clock::now();
            double dur = 1e-9 * std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
            out.time.push_back(dur);
        }

        solve_upper_triangular(k1, H, m1, eta);
        for (int k = 0; k < k1; ++k)
            cuddh::axpby(n, eta(k), V + k * n, one, x); // x <- x + eta[k] * v[k]

        A->action(x, r); // r <- A * x
        out.num_matvec++;
        cuddh::axpby(n, one, b, -one, r); // r <- b - r = b - A * x

        rnrm = cuddh::norm(n, r);
        out.res_norm.back() = (double)rnrm;

        if (rnrm <= opts.tol * bnrm + opts.atol)
        {
            out.success = true;
            break;
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        double dur = 1e-9 * std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
        out.time.back() = dur;
    }

    if (opts.verbose == SolverParams::ProgressBar)
        std::cout << std::endl;
    if (opts.verbose != SolverParams::Silent)
    {
        std::cout << "After " << it << " iterations (" << format_time(out.time.back())
                  << "), GMRES achieved rel. residual of " << out.res_norm.back() / bnrm << std::endl;
        if (out.success)
            std::cout << "GMRES successfully converged within desired tolerance." << std::endl;
        else
            std::cout << "GMRES failed to converge within desired tolerance." << std::endl;
    }

    out.num_iter = it;
    return out;
}

SolverResults cuddh::gmres(int n, double *x, const Operator *A, const double *b, SolverParams opts)
{
    return t_gmres<double>(n, x, A, b, opts);
}

SolverResults cuddh::gmres(int n, double *x, const Operator *A, const double *b, const Operator *P, SolverParams opts)
{
    PreconditionedSystem PA(n, A, P);

    host_device_dvec r0(n);
    double *d_r0 = r0.device_write();
    P->action(b, d_r0);

    return t_gmres<double>(n, x, &PA, d_r0, opts);
}

SolverResults cuddh::gmres(int n, float *x, const SinglePrecisionOperator *A, const float *b, SolverParams opts)
{
    return t_gmres<float>(n, x, A, b, opts);
}

cuddh::SolverResults cuddh::minres(int n, double *x, const Operator *A, const double *b, SolverParams opts)
{
    SolverResults out{.success = false, .num_iter = 0, .num_matvec = 0, .res_norm = {}, .time = {}};
    out.res_norm.reserve(opts.maxit + 1);
    out.time.reserve(opts.maxit + 1);

    thrust::device_vector<double> _r(n), _v(n), _w(n, 0.0), _wp(n, 0.0), _vp(n, 0.0), _wpp(n, 0.0);
    double *r = thrust::raw_pointer_cast(_r.data());
    double *v = thrust::raw_pointer_cast(_v.data());
    double *w = thrust::raw_pointer_cast(_w.data());
    double *wp = thrust::raw_pointer_cast(_wp.data());
    double *vp = thrust::raw_pointer_cast(_vp.data());
    double *wpp = thrust::raw_pointer_cast(_wpp.data());

    double bnrm = cuddh::norm(n, b);

    // r = b - A * x
    A->action(x, r);
    out.num_matvec++;
    axpby(n, 1.0, b, -1.0, r);

    double phi = norm(n, r);

    out.res_norm.push_back(phi);
    out.time.push_back(0.0);
    auto t0 = std::chrono::high_resolution_clock::now();

    if (phi < opts.tol * bnrm + opts.atol)
    {
        out.success = true;

        if (opts.verbose != SolverParams::Silent)
        {
            std::cout << "After 0 iterations, MINRES achieved rel. residual of " << out.res_norm.back() / bnrm
                      << std::endl;
            std::cout << "MINRES successfully converged within desired tolerance." << std::endl;
        }
        return out;
    }

    ProgressBar bar(opts.maxit);
    if (opts.verbose != SolverParams::Silent)
        std::cout << std::setprecision(5) << std::scientific;

    // v = r / phi
    axpby(n, 1.0 / phi, r, 0.0, v);

    double cp = 1.0, sp = 0.0;
    double c = 1.0, s = 0.0;
    double beta = 0.0;

    int it;
    for (it = 0; it < opts.maxit; ++it)
    {
        // Lanczos step
        A->action(v, r); // r = A * v
        out.num_matvec++;
        double alpha = dot(n, v, r); // (v, A*v)

        // r = A * v - alpha * v - beta * vp
        forall(n, [=] __device__(int i) { r[i] -= alpha * v[i] + beta * vp[i]; });

        // Givens
        double rho2 = sp * beta;
        double gamma = cp * beta;

        double rho1 = c * gamma + s * alpha;
        double delta = -s * gamma + c * alpha;

        beta = norm(n, r);

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

        out.res_norm.push_back(std::abs(phi));
        auto t1 = std::chrono::high_resolution_clock::now();
        double dur = 1e-9 * std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
        out.time.push_back(dur);
        if (opts.verbose == SolverParams::ProgressBar)
        {
            ++bar;
            std::cout << "[" << bar.get() << "] || iteration " << std::setw(10) << it + 1 << " / " << opts.maxit
                      << " || rel. res. = " << std::setw(10) << std::abs(phi) / bnrm << "\r" << std::flush;
        }
        else if (opts.verbose == SolverParams::Iteration)
        {
            std::cout << "iteration " << std::setw(10) << it + 1 << " / " << opts.maxit
                      << " || rel. res. = " << std::setw(10) << std::abs(phi) / bnrm << std::endl;
        }

        if (std::abs(phi) < opts.tol * bnrm + opts.atol)
        {
            out.success = true;
            break;
        }

        // prepare for next iteration
        forall(n, [=] __device__(int i) {
            wpp[i] = wp[i];
            wp[i] = w[i];
            vp[i] = v[i];
            v[i] = r[i] / beta;
        });
    }

    out.num_iter = it + 1;

    if (opts.verbose == SolverParams::ProgressBar)
        std::cout << std::endl;
    if (opts.verbose != SolverParams::Silent)
    {
        std::cout << "After " << out.num_iter << " iterations (" << format_time(out.time.back())
                  << "), MINRES achieved rel. residual of " << out.res_norm.back() / bnrm << std::endl;
        if (out.success)
            std::cout << "MINRES successfully converged in " << out.num_iter << " iterations." << std::endl;
        else
            std::cout << "MINRES reached maximum number of iterations (" << opts.maxit << ") without converging."
                      << std::endl;
    }

    return out;
}

SolverResults cuddh::fgmres(int n, double *x, const Operator *A, const double *b, const Operator *Precond,
                            SolverParams opts)
{
    validate_opts(opts);

    const double bnrm = cuddh::norm(n, b);

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

    SolverResults out;
    out.res_norm.reserve((opts.maxit + 1) * opts.m);
    out.time.reserve((opts.maxit + 1) * opts.m);
    out.num_matvec = 0;
    out.success = false;

    // compute initial residual
    A->action(x, r); // r <- A * x
    out.num_matvec++;
    axpby(n, 1.0, b, -1.0, r); // r <- b - A * x

    double rnrm = cuddh::norm(n, r);

    out.res_norm.push_back(rnrm);
    out.time.push_back(0.0);
    auto t0 = std::chrono::high_resolution_clock::now();

    if (rnrm <= opts.tol * bnrm + opts.atol)
    {
        out.success = true;

        if (opts.verbose != SolverParams::Silent)
        {
            std::cout << "After 0 iterations, F-GMRES achieved rel. residual of " << out.res_norm.back() / bnrm
                      << std::endl;
            std::cout << "F-GMRES successfully converged within desired tolerance." << std::endl;
        }

        return out;
    }

    ProgressBar bar(opts.maxit);
    if (opts.verbose != SolverParams::Silent)
        std::cout << std::setprecision(5) << std::scientific;

    int it = 1;
    while (it <= opts.maxit)
    {
        axpby(n, 1.0 / rnrm, r, 0.0, V); // v[0] <- r / ||r||
        eta(0) = rnrm;

        // Arnoldi process with variable preconditioner
        int k1 = 0;
        for (int k = 0; k < opts.m && it <= opts.maxit; ++k, ++it)
        {
            k1 = k + 1;
            const double *vk = V + k * n;
            double *vk1 = V + k1 * n;
            double *zk = Z + k * n;

            Precond->action(vk, zk); // z[k] <- Precond * v[k]
            A->action(zk, vk1);      // v[k+1] <- A * z[k]
            out.num_matvec++;

            // Modified Gram-Schmidt
            for (int j = 0; j <= k; ++j)
            {
                const double *vj = V + j * n;
                H(j, k) = cuddh::dot(n, vk1, vj);
                cuddh::axpby(n, -H(j, k), vj, 1.0, vk1); // v[k+1] <- v[k+1] - H(j, k) * v[j]
            }

            H(k + 1, k) = cuddh::norm(n, vk1);

            if (H(k + 1, k) < 1e-14)
                break;

            cuddh::scal(n, 1.0 / H(k + 1, k), vk1); // v[k+1] <- v[k+1] / ||v[k+1||

            givens_rotations(&H(0, k), (double *)cs, (double *)sn, k);
            eta(k + 1) = -sn(k) * eta(k);
            eta(k) = cs(k) * eta(k);

            rnrm = std::abs(eta(k + 1));
            out.res_norm.push_back(rnrm);

            if (opts.verbose == SolverParams::ProgressBar)
            {
                ++bar;
                std::cout << "[" << bar.get() << "] || iteration " << std::setw(10) << it << " / " << opts.maxit
                          << " || rel. res. = " << std::setw(10) << rnrm / bnrm << "\r" << std::flush;
            }
            else if (opts.verbose == SolverParams::Iteration)
            {
                std::cout << "iteration " << std::setw(10) << it << " / " << opts.maxit
                          << " || rel. res. = " << std::setw(10) << rnrm / bnrm << std::endl;
            }

            if (rnrm <= opts.tol * bnrm + opts.atol)
                break;

            auto t1 = std::chrono::high_resolution_clock::now();
            double dur = 1e-9 * std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
            out.time.push_back(dur);
        }

        solve_upper_triangular(k1, H, opts.m + 1, eta);
        for (int k = 0; k < k1; ++k)
            cuddh::axpby(n, eta(k), Z + k * n, 1.0, x); // x <- x + eta[k] * z[k]

        A->action(x, r); // r <- A * x
        out.num_matvec++;
        axpby(n, 1.0, b, -1.0, r); // r <- b - r = b - A * x

        rnrm = cuddh::norm(n, r);
        out.res_norm.back() = rnrm;

        if (rnrm <= opts.tol * bnrm + opts.atol)
        {
            out.success = true;
            break;
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        double dur = 1e-9 * std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
        out.time.back() = dur;
    }

    if (opts.verbose == SolverParams::ProgressBar)
        std::cout << std::endl;
    if (opts.verbose != SolverParams::Silent)
    {
        std::cout << "After " << it << " iterations (" << format_time(out.time.back())
                  << "), F-GMRES achieved rel. residual of " << out.res_norm.back() / bnrm << std::endl;
        if (out.success)
            std::cout << "F-GMRES successfully converged within desired tolerance." << std::endl;
        else
            std::cout << "F-GMRES failed to converge within desired tolerance." << std::endl;
    }

    out.num_iter = it;
    return out;
}
