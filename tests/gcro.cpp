#include "LinearSolvers/gcro.hpp"

#include <cstdlib>

#include "test_common.hpp"

using namespace cuddh;

namespace
{
    // 2D Finite Difference Advection-Diffusion Operator
    // -ε Δu + v.∇u = f
    // Square grid: n x n with ndof = n^2 total degrees of freedom
    // Linear indexing: global_index = i*n + j
    class AdvectionDiffusionOperator2D : public Operator
    {
    public:
        AdvectionDiffusionOperator2D(int n1d, double epsilon_ = 0.1, double vx_ = 1.0, double vy_ = 0.5)
            : n{n1d}, ndof{n * n}, epsilon{epsilon_}, vx{vx_}, vy{vy_}
        {}

        constexpr ~AdvectionDiffusionOperator2D() = default;

        void action(const double *x, double *y) const override
        {
            const double h = 1.0 / (n - 1);

            forall(ndof, [=, *this] __device__(int idx) -> void {
                int i = idx / n;
                int j = idx % n;

                if (i == 0 || i == n - 1 || j == 0 || j == n - 1)
                {
                    y[idx] = x[idx];
                    return;
                }

                int ip1 = (i + 1) * n + j;
                int im1 = (i - 1) * n + j;
                int jp1 = i * n + (j + 1);
                int jm1 = i * n + (j - 1);

                double laplacian = (x[ip1] + x[im1] + x[jp1] + x[jm1] - 4.0 * x[idx]) / (h * h);

                double du_dx = (x[jp1] - x[jm1]) / (2.0 * h);
                double du_dy = (x[ip1] - x[im1]) / (2.0 * h);
                double advection = vx * du_dx + vy * du_dy;

                y[idx] = -epsilon * laplacian + advection;
            });
        }

        void action(double c, const double *x, double *y) const override
        {
            cuddh_verify(false, printf("Not Implemented."));
        }

    private:
        int n, ndof;
        double epsilon, vx, vy;
    };

    class InexactPreconditioner : public Operator
    {
    public:
        InexactPreconditioner(int n_, const Operator &A_) : n{n_}, A{&A_} {}

        void action(const double *x, double *y) const override
        {
            zeros(n, y);
            gmres(n, y, A, x, {.m = 5, .maxit = 5, .tol = 1e-2, .atol = 0.0, .verbose = SolverVerbosity::Silent});
        }

        void action(double c, const double *x, double *y) const override
        {
            cuddh_verify(false, printf("Not Implemented."));
        }

    private:
        int n;
        const Operator *A;
    };
} // namespace

static void run_gcro_advection_diffusion_2d_test(TestLogger &summary)
{
    std::srand(1337);

    const int n = 32;
    const int ndof = n * n;

    AdvectionDiffusionOperator2D A(n, 0.1, 1.0, 0.5);

    thrust::universal_vector<double> _x_exact(ndof);
    thrust::universal_vector<double> _x(ndof);
    thrust::universal_vector<double> _b(ndof);
    thrust::universal_vector<double> _r(ndof);

    double *x_exact = thrust::raw_pointer_cast(_x_exact.data());
    double *x = thrust::raw_pointer_cast(_x.data());
    double *b = thrust::raw_pointer_cast(_b.data());
    double *r = thrust::raw_pointer_cast(_r.data());

    for (int i = 0; i < ndof; ++i)
        x_exact[i] = static_cast<double>(std::rand()) / RAND_MAX - 0.5;

    A.action(x_exact, b);
    zeros(ndof, x);

    // Test 1: WITHOUT preconditioner
    {
        zeros(ndof, x);
        const GCROParams opts = {
            .maxit = 1000,
            .rtol = 1e-6,
            .atol = 0.0,
            .verbose = SolverVerbosity::ProgressBar,
        };

        const SolverResults out = GCRO<double>(ndof, A, nullptr, 10, 5).solve(x, b, opts);

        A.action(x, r);
        axpby(ndof, 1.0, b, -1.0, r);

        const double b_norm = cuddh::norm(ndof, b);
        const double rel_res = cuddh::norm(ndof, r) / b_norm;
        const double target = opts.rtol + opts.atol / b_norm;

        if (out.success && rel_res <= target)
        {
            summary.pass("gcro 2D advection-diffusion WITHOUT preconditioner");
        }
        else
        {
            summary.fail("gcro 2D advection-diffusion WITHOUT preconditioner",
                         std::format("success={}, rel_res={}, target={}", out.success, rel_res, target));
        }
    }

    // Test 2: WITH GMRES(5) preconditioner
    {
        zeros(ndof, x);
        InexactPreconditioner M(ndof, A);
        const GCROParams opts = {
            .maxit = 100,
            .rtol = 1e-6,
            .atol = 0.0,
            .verbose = SolverVerbosity::ProgressBar,
        };

        const SolverResults out = GCRO<double>(ndof, A, &M, 10, 5).solve(x, b, opts);

        A.action(x, r);
        axpby(ndof, 1.0, b, -1.0, r);

        const double b_norm = cuddh::norm(ndof, b);
        const double rel_res = cuddh::norm(ndof, r) / b_norm;
        const double target = opts.rtol + opts.atol / b_norm;

        if (out.success && rel_res <= target)
        {
            summary.pass("gcro 2D advection-diffusion WITH GMRES(5) preconditioner");
        }
        else
        {
            summary.fail("gcro 2D advection-diffusion WITH GMRES(5) preconditioner",
                         std::format("success={}, rel_res={}, target={}", out.success, rel_res, target));
        }
    }
}

int main()
{
    TestLogger summary;
    run_gcro_advection_diffusion_2d_test(summary);
    return summary.finish();
}
