#include "gmres.hpp"

// lapack routine for triangular solve
extern "C" void dtrsv_(char * uplo, char * trans, char * diag, int * n, const double * a, int * lda, double * x, int * inc_x);
extern "C" void strsv_(char * uplo, char * trans, char * diag, int * n, const float * a, int * lda, float * x, int * inc_x);

template <typename scalar>
static void givens_rotations(scalar * h, scalar * cs, scalar * sn, int k)
{
    for (int i = 0; i < k; ++i)
    {
        const scalar h1 = h[i], h2 = h[i+1];
        h[i] = cs[i] * h1 + sn[i] * h2;
        h[i+1] = -sn[i] * h1 + cs[i] * h2;
    }

    scalar t = std::hypot(h[k], h[k+1]);
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

static void solve_upper_triangular(int N, const float * R, int ldr, float * b)
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
    class PreconditionedSystem : public Operator
    {
    public:
        PreconditionedSystem(int n, const Operator * A_, const Operator * P_) : q(n), A{A_}, P{P_} {}

        void action(double c, const double * x, double * y) const override
        {
            cuddh_error("How did you get this error?");
        }

        void action(const double * x, double * y) const override
        {
            double * d_q = q.device_write();
            A->action(x, d_q);
            P->action(d_q, y);
        }

    private:
        mutable host_device_dvec q;
        const Operator * A;
        const Operator * P;
    };

    template <typename scalar,typename OpType>
    solver_out t_gmres(int n, scalar * x, const OpType * A, const scalar * b, int m, int maxit, scalar tol, int verbose)
    {
        constexpr scalar one = 1, zero = 0;

        scalar bnrm = norm(n, b);

        const int m1 = m + 1;

        // DEVICE DATA:
        HostDeviceArray<scalar> _r(n);
        HostDeviceArray<scalar> _V(n * m1);
        scalar * r = _r.device_write();
        scalar * V = _V.device_write();

        // HOST DATA
        Matrix<scalar> H(m1, m);
        Vec<scalar> sn(m);
        Vec<scalar> cs(m);
        Vec<scalar> eta(m1);

        solver_out out;
        out.res_norm.reserve(maxit+1);
        out.time.reserve(maxit+1);
        out.num_matvec = 0;
        out.success = false;

        A->action(x, r); // r <- A * x
        out.num_matvec++;
        axpby(n, one, b, -one, r); // r <- b - r = b - A * x

        scalar r_nrm = norm(n, r);

        out.res_norm.push_back((double)r_nrm);
        out.time.push_back(0.0);
        auto t0 = std::chrono::high_resolution_clock::now();

        if (r_nrm < tol * bnrm)
        {
            out.success = true;
            
            if (verbose)
            {
                std::cout << "After 0 iterations, GMRES achieved rel. residual of " << out.res_norm.back()/bnrm << std::endl;
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
            scalar * vk = V;
            scalar * vk1;

            axpby(n, one/r_nrm, r, zero, vk); // v[0] <- r / ||r||

            // eta.zeros();
            std::fill(eta.begin(), eta.end(), 0.0);
            eta(0) = r_nrm;

            int k1=0;
            for (int k = 0; k < m; ++k)
            {
                k1 = k + 1;
                vk = V + k * n;
                vk1 = vk + n;

                A->action(vk, vk1); // v[k+1] <- A * v[k]
                out.num_matvec++;

                for (int j = 0; j < k1; ++j)
                {
                    const scalar * vj = V + j * n;
                    H(j, k) = dot(n, vk1, vj);
                    axpby(n, -H(j, k), vj, one, vk1); // v[k+1] <- v[k+1] - H(j, k) * v[j]
                }

                H(k1, k) = norm(n, vk1);

                if (H(k1, k) == 0.0)
                    break;

                scal(n, one / H(k1, k), vk1); // v[k+1] <- v[k+1] / ||v[k+1]||

                givens_rotations(&H(0, k), (scalar*)cs, (scalar*)sn, k);
                eta(k1) = -sn(k) * eta(k);
                eta(k) = cs(k) * eta(k);

                if (std::abs(eta(k1)) < tol * bnrm)
                    break;
            }

            solve_upper_triangular(k1, H, m1, eta);
            for (int k = 0; k < k1; ++k)
                axpby(n, eta(k), V + k*n, one, x); // x <- x + eta[k] * v[k]

            A->action(x, r); // r <- A * x
            out.num_matvec++;
            axpby(n, one, b, -one, r); // r <- b - r = b - A * x
            
            r_nrm = norm(n, r);
            out.res_norm.push_back((double)r_nrm);
            auto t1 = std::chrono::high_resolution_clock::now();
            double dur = 1e-9 * std::chrono::duration_cast<std::chrono::nanoseconds>(t1-t0).count();
            out.time.push_back(dur);

            if (verbose == 1)
            {
                ++bar;
                std::cout << "[" << bar.get() << "] || iteration " << std::setw(10) << it+1 << " / " << maxit << " || rel. res. = " << std::setw(10) << r_nrm/bnrm << "\r" << std::flush;
            }
            else if (verbose >= 2)
            {
                std::cout << "iteration " << std::setw(10) << it+1 << " / " << maxit << " || rel. res. = " << std::setw(10) << r_nrm/bnrm << std::endl;
            }

            if (r_nrm < tol * bnrm)
            {
                out.success = true;
                break;
            }
        }

        if (verbose == 1)
            std::cout << std::endl;
        if (verbose)
        {
            std::cout << "After " << it << " iterations, GMRES achieved rel. residual of " << out.res_norm.back()/bnrm << std::endl;
            if (out.success)
                std::cout << "GMRES successfully converged within desired tolerance." << std::endl;
            else
                std::cout << "GMRES failed to converge within desired tolerance." << std::endl;
        }

        out.num_iter = it;
        return out;
    }

    solver_out gmres(int n, double * x, const Operator * A, const double * b, int m, int maxit, double tol, int verbose)
    {
        return t_gmres<double>(n, x, A, b, m, maxit, tol, verbose);
    }

    solver_out gmres(int n, double * x, const Operator * A, const double * b, const Operator * P, int m, int maxit, double tol, int verbose)
    {
        PreconditionedSystem PA(n, A, P);
        
        host_device_dvec r0(n);
        double * d_r0 = r0.device_write();
        P->action(b, d_r0);

        return t_gmres<double>(n, x, &PA, d_r0, m, maxit, tol, verbose);
    }

    solver_out gmres(int n, float * x, const SinglePrecisionOperator * A, const float * b, int m, int maxit, float tol, int verbose)
    {
        return t_gmres<float>(n, x, A, b, m, maxit, tol, verbose);
    }
} // namespace cuddh
