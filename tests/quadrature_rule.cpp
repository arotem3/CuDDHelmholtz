#include "test.hpp"

// n-th Chebyshev polynomial of first kind
double Tn(double x, int n)
{
    double z = std::acos(x);
    return std::cos(n * z);
}

// f(x) is a polynomial of degree n with integral of 2 over [-1, 1]
double f(double x, int n)
{
    const double a = 1.0 - n*n;
    const double b = 1.0 - (n-1.0)*(n-1.0);
    return a * Tn(x, n) + b * Tn(x, n-1);
}

using namespace cuddh;

namespace cuddh_test
{
    int t_quadrature_rule()
    {
        for (int n=1; n < 16; ++n)
        {
            QuadratureRule q(n, QuadratureRule::GaussLegendre);
            const int p = 2*n-1; // Gauss-Legendre rule is exact for polynomials of degree 2*n-1;

            double I = 0.0;
            for (int i=0; i < n; ++i)
            {
                I += q.w()[i] * f(q.x()[i], p);
            }

            const double error = std::abs(I - 2.0);

            if (error > 1e-10)
            {
                std::cout << "\tt_quadrature_rule(): Gauss Legendre with n = " << n << " points computed incorrect integral.\n";
                return 0;
            }
        }

        for (int n=2; n < 16; ++n)
        {
            QuadratureRule q(n, QuadratureRule::GaussLobatto);
            const int p = 2*n-3; // Gauss-Legendre rule is exact for polynomials of degree 2*n-1;

            double I = 0.0;
            for (int i=0; i < n; ++i)
            {
                I += q.w()[i] * f(q.x()[i], p);
            }

            const double error = std::abs(I - 2.0);

            if (error > 1e-10)
            {
                std::cout << "\tt_quadrature_rule(): Gauss Lobatto with n = " << n << " points computed incorrect integral.\n";
                return 0;
            }
        }

        return 1;
    }
} // namespace cuddh_test
