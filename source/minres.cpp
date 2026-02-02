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

    double beta = norm(n, r);
    double phi_bar = beta; // ||r||_2

    out.res_norm.push_back(phi_bar);
    out.time.push_back(0.0);
    auto t0 = std::chrono::high_resolution_clock::now();

    if (phi_bar < tol * bnrm)
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

    // v = r / beta
    axpby(n, 1.0 / beta, r, 0.0, v);

    double cp = 1.0, sp = 0.0;
    double c = 1.0, s = 0.0;
    double beta_qr = 0.0;

    int it;
    for (it = 0; it < maxit; ++it)
    {
        // Lanczos step
        A->action(v, r); // r = A * v
        out.num_matvec++;
        double alpha = dot(n, v, r); // (v, A*v)

        // r = A * v - alpha * v - beta * vp
        axpby(n, -alpha, v, 1.0, r);
        axpby(n, -beta, vp, 1.0, r);

        double beta_next = norm(n, r);

        // Givens
        double rho2 = sp * beta_qr;
        double gamma = cp * beta_qr;

        double rho1 = c * gamma + s * alpha;
        double delta = -s * gamma + c * alpha;

        double rho3 = std::hypot(delta, beta_next);
        cp = c;
        sp = s;
        c = delta / rho3;
        s = beta_next / rho3;

        // update w
        // w = (v - rho1 * wp - rho2 * wpp) / rho3
        axpby(n, 1.0 / rho3, v, 0.0, w);
        axpby(n, -rho1 / rho3, wp, 1.0, w);
        axpby(n, -rho2 / rho3, wpp, 1.0, w);

        // x = x + (phi_bar * c) * w
        axpby(n, phi_bar * c, w, 1.0, x);

        // check convergence
        phi_bar = -s * phi_bar;

        out.res_norm.push_back(std::abs(phi_bar));
        auto t1 = std::chrono::high_resolution_clock::now();
        double dur = 1e-9 * std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
        out.time.push_back(dur);
        if (verbose == 1)
        {
            ++bar;
            std::cout << "[" << bar.get() << "] || iteration " << std::setw(10) << it + 1 << " / " << maxit
                      << " || rel. res. = " << std::setw(10) << std::abs(phi_bar) / bnrm << "\r" << std::flush;
        }
        else if (verbose >= 2)
        {
            std::cout << "iteration " << std::setw(10) << it + 1 << " / " << maxit
                      << " || rel. res. = " << std::setw(10) << std::abs(phi_bar) / bnrm << std::endl;
        }

        if (std::abs(phi_bar) < tol * bnrm)
        {
            out.success = true;
            break;
        }

        // prepare for next iteration
        copy(n, wp, wpp);
        copy(n, w, wp);
        copy(n, v, vp);

        axpby(n, 1.0 / beta_next, r, 0.0, v);
        beta = beta_next;
        beta_qr = beta_next;
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