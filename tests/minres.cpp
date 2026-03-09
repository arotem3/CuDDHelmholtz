#include "test_common.hpp"

using namespace cuddh;

static void run_minres_test(TestLogger &summary)
{
    auto a = sym_test_mat<double>();
    const int n = a.size();

    host_device_dvec _x(n);

    dvec_wrapper h_x(_x.host_write(), n);
    for (int i = 0; i < n; ++i)
        h_x(i) = static_cast<double>(rand()) / RAND_MAX;

    host_device_dvec _y(n);
    double *y = _y.device_write();

    double *x = _x.device_read_write();

    a.action(x, y); // y <- A * random

    dla::zeros(n, x);

    const SolverParams opts = {.maxit = n, .rtol = 1e-10, .atol = 0.0, .verbose = SolverParams::ProgressBar};
    auto out = cuddh::minres(n, x, a, y, opts);

    host_device_dvec _r(n);
    double *r = _r.device_write();
    a.action(x, r);
    dla::axpby(n, 1.0, y, -1.0, r);

    const double b_norm = dla::norm(n, y);
    const double rel_res = dla::norm(n, r) / b_norm;
    const double target = opts.rtol + opts.atol / b_norm;

    if (out.success && rel_res <= target)
    {
        summary.pass("minres solve");
    }
    else
    {
        summary.fail("minres solve", std::format("success={}, rel_res={}, target={}", out.success, rel_res, target));
    }
}

int main()
{
    TestLogger summary;
    run_minres_test(summary);
    return summary.finish();
}
