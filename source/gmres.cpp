#include "gmres.hpp"

// lapack routine for triangular solve
extern "C" void dtrsv_(char * uplo, char * trans, char * diag, int * n, const double * a, int * lda, double * x, int * inc_x);

static void givens_rotations(double * h, double * cs, double * sn, int k)
{
    for (int i = 0; i < k; ++i)
    {
        const double h1 = h[i], h2 = h[i+1];
        h[i] = cs[i] * h1 + sn[i] * h2;
        h[i+1] = -sn[i] * h1 + cs[i] * h2;
    }

    double t = std::hypot(h[k], h[k+1]);
    cs[k] = h[k]/t;
    sn[k] = h[k+1]/t;

    h[k] = cs[k]*h[k] + sn[k]*h[k+1];
    h[k+1] = 0.0;
}

// solve R * x == b inplace on b where R is upper triangular
static void solve_upper_triangular(int N, const double * R, int ldr, double * b)
{
    int i_one = 1.0;
    char uplo[] = "u";
    char trans[] = "n";
    char diag[] = "n";
    dtrsv_(uplo, trans, diag, &N, R, &ldr, b, &i_one);
}

class ProgressBar
{
public:
    ProgressBar(int nt_) : it{0}, nt{nt_}, progress(30, ' ') {}

    void operator++()
    {
        it = std::min(it+1, nt-1);
        progress.at(30*(it-1)/nt) = '#';
    }

    const std::string& get() const
    {
        return progress;
    }

private:
    int it;
    const int nt;
    std::string progress;
};

namespace cuddh
{
    solver_out gmres(int n, double * x, const Operator * A, const double * b, const Operator * Precond, int m, int maxit, double tol, int verbose)
    {
        const double one = 1.0, zero = 0.0;

        double bnrm = norm(n, b);

        const int m1 = m + 1;

        dvec r(n);
        dmat H(m1, m);
        dvec sn(m);
        dvec cs(m);
        dvec q(n);
        dmat V(n, m1);
        dvec eta(m1);

        solver_out out;
        out.res_norm.reserve(maxit+1);
        out.num_matvec = 0;
        out.num_precond = 0;
        out.success = false;

        A->action(x, q); // q <- A * x
        out.num_matvec++;
        axpby(n, one, b, -one, q); // q <- b - q = b - A * x

        Precond->action(q, r); // r <- M \ q = M \ (b - A * x)
        out.num_precond++;

        out.res_norm.push_back(norm(n, r));

        if (out.res_norm.at(0) < bnrm * tol)
        {
            out.num_iter = 0;
            out.success = true;

            if (verbose)
            {
                std::cout << "After 0 iterations, GMRES achieved rel. residual of " << out.res_norm.back() << std::endl;
                std::cout << "GMRES successfully converged within desired tolerance." << std::endl;
            }

            return out;
        }
        
        ProgressBar bar(maxit);
        if (verbose)
            std::cout << std::setprecision(5) << std::scientific;

        int it = 1;
        for (; it < maxit; ++it)
        {
            double * vk = V;
            double * vk1;

            const double r_nrm = out.res_norm.at(it-1);
            axpby(n, one/r_nrm, r, zero, vk); // v[0] <- r / ||r||

            eta.zeros();
            eta(0) = r_nrm;

            int k1;
            for (int k = 0; k < m; ++k)
            {
                k1 = k + 1;
                vk = V.data() + k * n;
                vk1 = vk + n;

                A->action(vk, q); // q <- A * v[k]
                out.num_matvec++;

                Precond->action(q, vk1); // v[k+1] <- M \ q = M \ (A * v[k])
                out.num_precond++;

                for (int j = 0; j < k1; ++j)
                {
                    const double * vj = V.data() + j * n;
                    H(j, k) = dot(n, vk1, vj);
                    axpby(n, -H(j, k), vj, one, vk1); // v[k+1] <- v[k+1] - H(j, k) * v[j]
                }

                H(k1, k) = norm(n, vk1);

                if (H(k1, k) == 0.0)
                    break;

                scal(n, one / H(k1, k), vk1); // v[k+1] <- v[k+1] / ||v[k+1]||

                givens_rotations(&H(0, k), cs, sn, k);
                eta(k1) = -sn(k) * eta(k);
                eta(k) = cs(k) * eta(k);

                if (std::abs(eta(k1)) < tol * bnrm)
                    break;
            }

            solve_upper_triangular(k1, H, m1, eta);
            for (int k = 0; k < k1; ++k)
                axpby(n, eta(k), V.data() + k*n, one, x); // x <- x + eta[k] * v[k]

            A->action(x, q); // q <- A * x
            out.num_matvec++;
            axpby(n, one, b, -one, q); // q <- b - q = b - A * x

            Precond->action(q, r); // r <- M \ q = M \ (b - A * x)
            out.num_precond++;
            
            out.res_norm.push_back(norm(n, r));

            if (verbose == 1)
            {
                ++bar;
                std::cout << "[" << bar.get() << "] || iteration " << std::setw(10) << it << " / " << maxit << " || rel. res. = " << std::setw(10) << out.res_norm.back() << "\r" << std::flush;
            }
            else if (verbose >= 2)
            {
                std::cout << "iteration " << std::setw(10) << it << " / " << maxit << " || rel. res. = " << std::setw(10) << out.res_norm.back() << std::endl;
            }

            if (out.res_norm.at(it) < tol * bnrm)
            {
                out.success = true;
                break;
            }
        }

        if (verbose == 1)
            std::cout << std::endl;
        if (verbose)
        {
            std::cout << "After " << it << " iterations, GMRES achieved rel. residual of " << out.res_norm.back() << std::endl;
            if (out.success)
                std::cout << "GMRES successfully converged within desired tolerance." << std::endl;
            else
                std::cout << "GMRES failed to converge within desired tolerance." << std::endl;
        }

        out.num_iter = it;
        return out;
    }
} // namespace cuddh
