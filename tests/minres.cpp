#include "test_common.hpp"

using namespace cuddh;

template <typename real_t>
static void run_minres_test(TestLogger &summary, std::string_view precision, double rtol)
{
    auto a = sym_test_mat<real_t>();
    const int n = a.size();

    thrust::universal_vector<real_t> _x_exact(n);
    thrust::device_vector<real_t> _x(n);
    thrust::device_vector<real_t> _b(n);
    thrust::device_vector<real_t> _r(n);

    real_t *x_exact = thrust::raw_pointer_cast(_x_exact.data());
    real_t *x = thrust::raw_pointer_cast(_x.data());
    real_t *b = thrust::raw_pointer_cast(_b.data());
    real_t *r = thrust::raw_pointer_cast(_r.data());

    for (int i = 0; i < n; ++i)
        x_exact[i] = static_cast<real_t>(std::rand()) / static_cast<real_t>(RAND_MAX);

    a.action(x_exact, b); // b <- A * random

    dla::zeros(n, x);

    const SolverParams opts = {.maxit = n, .rtol = rtol, .atol = 0.0, .verbose = SolverParams::ProgressBar};
    auto out = cuddh::minres(n, x, a, b, opts);

    a.action(x, r);
    dla::axpby(n, static_cast<real_t>(1.0), b, static_cast<real_t>(-1.0), r);

    const real_t b_norm = dla::norm(n, b);
    const real_t rel_res = dla::norm(n, r) / b_norm;
    const real_t target = static_cast<real_t>(opts.rtol + opts.atol / b_norm);

    const auto test_name = std::format("minres solve ({})", precision);

    if (out.success && rel_res <= target)
    {
        summary.pass(test_name);
    }
    else
    {
        summary.fail(test_name, std::format("success={}, rel_res={}, target={}", out.success, rel_res, target));
    }
}

int main()
{
    TestLogger summary;
    run_minres_test<double>(summary, "double", 1e-10);
    run_minres_test<float>(summary, "float", 1e-5);
    return summary.finish();
}
