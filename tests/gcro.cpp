#include "test_common.hpp"

using namespace cuddh;

static void run_gcro_test(TestLogger &summary)
{
    std::srand(1337);

    auto A = asym_test_mat<double>();
    const int n = A.size();

    thrust::universal_vector<double> _x_exact(n);
    thrust::device_vector<double> _x(n);
    thrust::device_vector<double> _b(n);
    thrust::device_vector<double> _r(n);

    double *x_exact = thrust::raw_pointer_cast(_x_exact.data());
    double *x = thrust::raw_pointer_cast(_x.data());
    double *b = thrust::raw_pointer_cast(_b.data());
    double *r = thrust::raw_pointer_cast(_r.data());

    for (int i = 0; i < n; ++i)
        x_exact[i] = static_cast<double>(std::rand()) / RAND_MAX - 0.5;

    A.action(x_exact, b);
    dla::zeros(n, x);

    // Test 1: WITHOUT preconditioner
    {
        dla::zeros(n, x);
        const SolverParams opts = {
            .maxit = 1000,
            .rtol = 1e-6,
            .atol = 0.0,
            .verbose = SolverParams::ProgressBar,
        };

        const SolverResults out = GCRO<double>(n, A, nullptr, 10, 5).solve(x, b, opts);

        A.action(x, r);
        dla::axpby(n, 1.0, b, -1.0, r);

        const double b_norm = dla::norm(n, b);
        const double rel_res = dla::norm(n, r) / b_norm;
        const double target = opts.rtol + opts.atol / b_norm;

        if (out.success && rel_res <= target)
        {
            summary.pass("gcro solve WITHOUT preconditioner");
        }
        else
        {
            summary.fail("gcro solve WITHOUT preconditioner",
                         std::format("success={}, rel_res={}, target={}", out.success, rel_res, target));
        }
    }

    // Test 2: WITH inexact preconditioner
    {
        dla::zeros(n, x);
        InexactPreconditioner<double> M(n, A);
        const SolverParams opts = {
            .maxit = 100,
            .rtol = 1e-6,
            .atol = 0.0,
            .verbose = SolverParams::ProgressBar,
        };

        const SolverResults out = GCRO<double>(n, A, &M, 10, 5).solve(x, b, opts);

        A.action(x, r);
        dla::axpby(n, 1.0, b, -1.0, r);

        const double b_norm = dla::norm(n, b);
        const double rel_res = dla::norm(n, r) / b_norm;
        const double target = opts.rtol + opts.atol / b_norm;

        if (out.success && rel_res <= target)
        {
            summary.pass("gcro solve WITH inexact preconditioner");
        }
        else
        {
            summary.fail("gcro solve WITH inexact preconditioner",
                         std::format("success={}, rel_res={}, target={}", out.success, rel_res, target));
        }
    }
}

int main()
{
    TestLogger summary;
    run_gcro_test(summary);
    return summary.finish();
}
