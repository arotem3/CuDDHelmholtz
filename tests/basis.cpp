#include <cmath>

#include "test_common.hpp"

using namespace cuddh;

static double jacobiP_next(unsigned int m, double a, double b, double x, double y1, double y2)
{
    double yp1 = (2 * m + a + b - 1) * ((2 * m + a + b) * (2 * m + a + b - 2) * x + a * a - b * b) * y1 -
                 2 * (m + a - 1) * (m + b - 1) * (2 * m + a + b) * y2;
    yp1 /= 2 * m * (m + a + b) * (2 * m + a + b - 2);
    return yp1;
}

static double jacobiP(unsigned int n, double a, double b, double x)
{
    double ym1 = 1;

    if (n == 0)
        return ym1;

    double y = (a + 1) + 0.5 * (a + b + 2) * (x - 1);

    for (unsigned int m = 2; m <= n; ++m)
    {
        double yp1 = jacobiP_next(m, a, b, x, y, ym1);
        ym1 = y;
        y = yp1;
    }

    return y;
}

static double jacobiP_derivative(unsigned int k, unsigned int n, double a, double b, double x)
{
    if (k > n)
        return 0.0;
    else
    {
        double s = std::lgamma(n + a + b + 1 + k) - std::lgamma(n + a + b + 1) - k * std::log(2);
        return std::exp(s) * jacobiP(n - k, a + k, b + k, x);
    }
}

static void run_basis_tests(TestLogger &summary)
{
    for (int n = 2; n < 15; ++n)
    {
        Basis b(n);

        dvec y(n);
        auto q = b.quadrature().x(MemorySpace::HOST);
        for (int i = 0; i < n; ++i)
            y(i) = jacobiP(n - 1, 0, 0, q(i));

        const int m = 10;
        dvec x(m);
        for (int i = 0; i < m; ++i)
            x(i) = -1.0 + 2.0 * i / (m - 1);

        dmat D(m, n);
        b.deriv(m, x, D);

        double max_error = 0.0;
        for (int i = 0; i < m; ++i)
        {
            double dy = 0.0;
            for (int j = 0; j < n; ++j)
            {
                dy += D(i, j) * y(j);
            }

            const double error = dy - jacobiP_derivative(1, n - 1, 0, 0, x(i));
            max_error = std::max(max_error, std::abs(error));
        }

        const auto deriv_name = std::format("basis derivative n={}", n);
        if (max_error < 1e-10)
            summary.pass(deriv_name);
        else
            summary.fail(deriv_name, std::format("max error {} exceeds tolerance", max_error));

        dmat P(m, n);
        b.eval(m, x, P);

        max_error = 0.0;
        for (int i = 0; i < m; ++i)
        {
            double p = 0.0;
            for (int j = 0; j < n; ++j)
            {
                p += P(i, j) * y(j);
            }

            const double error = p - jacobiP(n - 1, 0, 0, x(i));
            max_error = std::max(max_error, std::abs(error));
        }

        const auto eval_name = std::format("basis evaluation n={}", n);
        if (max_error < 1e-10)
            summary.pass(eval_name);
        else
            summary.fail(eval_name, std::format("max error {} exceeds tolerance", max_error));
    }
}

int main()
{
    TestLogger summary;
    run_basis_tests(summary);
    return summary.finish();
}
