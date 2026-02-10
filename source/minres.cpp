#include "minres.hpp"

class ProgressBar
{
public:
    ProgressBar(int nt_) : it{0}, nt{nt_}, progress(30, ' ') {}

    void operator++()
    {
        it = std::min(it + 1, nt - 1);
        progress.at(30 * (it - 1) / nt) = '#';
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

cuddh::solver_out cuddh::minres(int n, double *x, const Operator *A, const double *b, int maxit, double tol,
                                int verbose)
{
    solver_out out{.success = false, .num_iter = 0, .num_matvec = 0, .res_norm = {}, .time = {}};
    out.res_norm.reserve(maxit + 1);
    out.time.reserve(maxit + 1);

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

    if (phi < tol * bnrm)
    {
        out.success = true;

        if (verbose)
        {
            std::cout << "After 0 iterations, MINRES achieved rel. residual of " << out.res_norm.back() / bnrm
                      << std::endl;
            std::cout << "MINRES successfully converged within desired tolerance." << std::endl;
        }
        return out;
    }

    ProgressBar bar(maxit);
    if (verbose)
        std::cout << std::setprecision(5) << std::scientific;

    // v = r / phi
    axpby(n, 1.0 / phi, r, 0.0, v);

    double cp = 1.0, sp = 0.0;
    double c = 1.0, s = 0.0;
    double beta = 0.0;

    int it;
    for (it = 0; it < maxit; ++it)
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
        if (verbose == 1)
        {
            ++bar;
            std::cout << "[" << bar.get() << "] || iteration " << std::setw(10) << it + 1 << " / " << maxit
                      << " || rel. res. = " << std::setw(10) << std::abs(phi) / bnrm << "\r" << std::flush;
        }
        else if (verbose >= 2)
        {
            std::cout << "iteration " << std::setw(10) << it + 1 << " / " << maxit
                      << " || rel. res. = " << std::setw(10) << std::abs(phi) / bnrm << std::endl;
        }

        if (std::abs(phi) < tol * bnrm)
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

    if (verbose == 1)
        std::cout << std::endl;
    if (verbose)
    {
        std::cout << "After " << out.num_iter << " iterations, MINRES achieved rel. residual of "
                  << out.res_norm.back() / bnrm << std::endl;
        if (out.success)
            std::cout << "MINRES successfully converged in " << out.num_iter << " iterations." << std::endl;
        else
            std::cout << "MINRES reached maximum number of iterations (" << maxit << ") without converging."
                      << std::endl;
    }

    return out;
}