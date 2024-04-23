#include "test.hpp"

/// @brief non-symmetric tridiagonal toeplitz matrix
class TestMatrix : public cuddh::Operator
{
public:
    TestMatrix(int n_) : n{n_} {}
    ~TestMatrix() = default;

    void action(const double * x, double * y) const override
    {
        constexpr double c[] = {1.0, -3.0, 1.5};

        y[0] = c[1] * x[0] + c[2] * x[1];
        for (int i = 1; i < n-1; ++i)
        {
            y[i] = c[0] * x[i-1] + c[1] * x[i] + c[2] * x[i+1];
        }
        y[n-1] = c[0] * x[n-2] + c[1] * x[n-1];
    }

    void action(double c, const double * x, double * y) const override
    {
        // not needed
    }

private:
    int n;
};

namespace cuddh_test
{
    void t_gmres(int& n_test, int& n_passed)
    {
        const int n = 1000;

        cuddh::dvec x(n);
        for (int i = 0; i < n; ++i)
            x(i) = (double)rand() / RAND_MAX;

        cuddh::dvec y(n);

        TestMatrix a(n);
        a.action(x, y);

        // x.zeros();
        zeros(x.size(), x);

        cuddh::Identity P(n);

        const int m = 50;
        const int maxit = 100;
        const double tol = 1e-10;
        auto out = cuddh::gmres(n, x, &a, y, &P, m, maxit, tol);

        n_test++;
        if (not out.success)
            std::cout << "\tt_gmres(): gmres failed to solve linear system. Final residual = " << out.res_norm.back() << "\n";
        else
            n_passed++;
    }
} // namespace cuddh_test
