#include <cmath>

#include "test_common.hpp"

// n-th Chebyshev polynomial of first kind
static double Tn(double x, int n)
{
    double z = std::acos(x);
    return std::cos(n * z);
}

// f(x) is a polynomial of degree n with integral of 2 over [-1, 1]
static double f(double x, int n)
{
    const double a = 1.0 - n * n;
    const double b = 1.0 - (n - 1.0) * (n - 1.0);
    return a * Tn(x, n) + b * Tn(x, n - 1);
}

using namespace cuddh;

static void run_quadrature_rule_tests(TestLogger &summary)
{
    for (int n = 1; n < 16; ++n)
    {
        QuadratureRule q(n, QuadratureRule::GaussLegendre);
        const int p = 2 * n - 1;

        double I = 0.0;
        for (int i = 0; i < n; ++i)
        {
            I += q.w(i) * f(q.x(i), p);
        }

        const double error = std::abs(I - 2.0);
        const auto test_name = std::format("quadrature_rule {} n={} p={}", q.name(), n, p);

        if (error < 1e-10)
            summary.pass(test_name);
        else
            summary.fail(test_name, std::format("quadrature error {} exceeds tolerance", error));
    }

    for (int n = 2; n < 16; ++n)
    {
        QuadratureRule q(n, QuadratureRule::GaussLobatto);
        const int p = 2 * n - 3;

        double I = 0.0;
        for (int i = 0; i < n; ++i)
        {
            I += q.w(i) * f(q.x(i), p);
        }

        const double error = std::abs(I - 2.0);
        const auto test_name = std::format("quadrature_rule {} n={} p={}", q.name(), n, p);

        if (error < 1e-10)
            summary.pass(test_name);
        else
            summary.fail(test_name, std::format("quadrature error {} exceeds tolerance", error));
    }
}

int main()
{
    TestLogger summary;
    run_quadrature_rule_tests(summary);
    return summary.finish();
}
