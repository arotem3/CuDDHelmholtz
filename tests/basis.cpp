#include "test.hpp"

using namespace cuddh;

static double jacobiP_next(unsigned int m, double a, double b, double x, double y1, double y2)
{
    double yp1 = (2*m + a + b - 1)*((2*m + a + b)*(2*m + a + b - 2)*x + a*a - b*b)*y1
                    - 2*(m + a - 1)*(m + b - 1)*(2*m + a + b)*y2;
    yp1 /= 2*m*(m + a + b)*(2*m + a + b - 2);
    return yp1;
}

static double jacobiP(unsigned int n, double a, double b, double x)
{
    double ym1 = 1;
    
    if (n == 0)
        return ym1;
    
    double y = (a + 1) + 0.5*(a + b + 2)*(x-1);
    
    for (unsigned int m=2; m <= n; ++m)
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
        double s = std::lgamma(n+a+b+1+k) - std::lgamma(n+a+b+1) - k * std::log(2);
        return std::exp(s) * jacobiP(n-k, a+k, b+k, x);
    }
}

namespace cuddh_test
{
    void t_basis(int& n_tests, int& n_passed)
    {
        for (int n=2; n < 15; ++n)
        {
            Basis b(n);

            dvec y(n);
            for (int i=0; i < n; ++i)
                y(i) = jacobiP(n-1, 0, 0, b.quadrature().x(i));

            const int m = 10;
            dvec x(m);
            for (int i=0; i < m; ++i)
                x(i) = -1.0 + 2.0*i/(m-1);

            dmat D(m, n);
            b.deriv(m, x, D);

            double max_error = 0.0;
            for (int i=0; i < m; ++i)
            {
                double dy = 0.0;
                for (int j = 0; j < n; ++j)
                {
                    dy += D(i, j) * y(j);
                }

                const double error = dy - jacobiP_derivative(1, n-1, 0, 0, x(i));
                max_error = std::max(max_error, std::abs(error));
            }

            n_tests++;
            if (max_error < 1e-10)
            {
                std::cout << "\t[ + ] t_basis(" << n << ") derivative test successful." << std::endl;
                n_passed++;
            }
            else
            {
                std::cout << "\t[ - ] t_basis(" << n << ") derivative test failed.\n\t\tComputed derivative had error ~ " << max_error << "but should have been exact to machine prec." << std::endl;
            }
            
            dmat P(m, n);
            b.eval(m, x, P);

            max_error = 0.0;
            for (int i=0; i < m; ++i)
            {
                double p = 0.0;
                for (int j=0; j < n; ++j)
                {
                    p += P(i, j) * y(j);
                }

                const double error = p - jacobiP(n-1, 0, 0, x(i));
                max_error = std::max(max_error, std::abs(error));
            }

            n_tests++;
            if (max_error < 1e-10)
            {
                std::cout << "\t[ + ] t_basis(" << n << ") evaluation test successful." << std::endl;
                n_passed++;
            }
            else
            {
                std::cout << "\t[ - ] t_basis(" << n << ") evaluation test failed.\n\t\tInterpolation had error ~ " << max_error << "but should have been exact to machine prec." << std::endl;
            }
        }
    }
} // namespace cuddh_test
