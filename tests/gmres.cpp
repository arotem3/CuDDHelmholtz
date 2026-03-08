#include <cstdlib>

#include "test_common.hpp"

using namespace cuddh;

namespace
{
    /// @brief non-symmetric tridiagonal toeplitz matrix
    class TestMatrix : public cuddh::Operator
    {
    public:
        TestMatrix(int n_) : _n{n_} {}
        ~TestMatrix() = default;

        void action(const double *x, double *y) const override
        {
            const int n = _n;
            forall(n, [=] __device__(int i) -> void {
                constexpr double c[] = {1.0, -3.0, 1.5};

                if (i == 0)
                    y[0] = c[1] * x[0] + c[2] * x[1];
                else if (i == n - 1)
                    y[n - 1] = c[0] * x[n - 2] + c[1] * x[n - 1];
                else
                    y[i] = c[0] * x[i - 1] + c[1] * x[i] + c[2] * x[i + 1];
            });
        }

        void action(double c, const double *x, double *y) const override
        {
            // not needed
        }

    private:
        int _n;
    };
} // namespace

static void run_gmres_test(TestLogger &summary)
{
    const int n = 1 << 10;

    host_device_dvec _x(n);

    dvec_wrapper h_x(_x.host_write(), n);
    for (int i = 0; i < n; ++i)
        h_x(i) = static_cast<double>(rand()) / RAND_MAX;

    host_device_dvec _y(n);
    double *y = _y.device_write();

    double *x = _x.device_read_write();

    TestMatrix a(n);
    a.action(x, y); // y <- A * random

    zeros(n, x);

    const gmresParams opts = {
        .m = 20, .maxit = 200, .tol = 1e-10, .atol = 0.0, .verbose = SolverVerbosity::ProgressBar};
    auto out = cuddh::gmres(n, x, &a, y, opts);

    host_device_dvec _r(n);
    double *r = _r.device_write();
    a.action(x, r);
    axpby(n, 1.0, y, -1.0, r);

    const double b_norm = cuddh::norm(n, y);
    const double rel_res = cuddh::norm(n, r) / b_norm;
    const double target = opts.tol + opts.atol / b_norm;

    if (out.success && rel_res <= target)
    {
        summary.pass("gmres tridiagonal solve");
    }
    else
    {
        summary.fail("gmres tridiagonal solve",
                     std::format("success={}, rel_res={}, target={}", out.success, rel_res, target));
    }
}

int main()
{
    TestLogger summary;
    run_gmres_test(summary);
    return summary.finish();
}
