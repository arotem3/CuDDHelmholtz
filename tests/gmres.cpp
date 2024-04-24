#include "test.hpp"

using namespace cuddh;

/// @brief non-symmetric tridiagonal toeplitz matrix
class TestMatrix : public cuddh::Operator
{
public:
    TestMatrix(int n_) : _n{n_} {}
    ~TestMatrix() = default;

    void action(const double * x, double * y) const override
    {
        const int n = _n;
        forall(n, [=] __device__ (int i) -> void {
            constexpr double c[] = {1.0, -3.0, 1.5};

            if (i == 0)
                y[0] = c[1] * x[0] + c[2] * x[1];
            else if (i == n-1)
                y[n-1] = c[0] * x[n-2] + c[1] * x[n-1];
            else
                y[i] = c[0] * x[i-1] + c[1] * x[i] + c[2] * x[i+1];
        });
    }

    void action(double c, const double * x, double * y) const override
    {
        // not needed
    }

private:
    int _n;
};

namespace cuddh_test
{
    void t_gmres(int& n_test, int& n_passed)
    {
        const int n = 1<<10;

        host_device_dvec _x(n);

        dvec_wrapper h_x(_x.host_write(), n);
        for (int i = 0; i < n; ++i)
            h_x(i) = (double)rand() / RAND_MAX;

        host_device_dvec _y(n);
        double * y = _y.device_write();

        double * x = _x.device_read_write();

        TestMatrix a(n);
        a.action(x, y); // y <- A * random

        zeros(n, x);

        const int m = 5;
        const int maxit = 100;
        const double tol = 1e-10;
        auto out = cuddh::gmres(n, x, &a, y, m, maxit, tol);

        n_test++;
        if (not out.success)
            std::cout << "\tt_gmres(): gmres failed to solve linear system. Final residual = " << out.res_norm.back() << "\n";
        else
            n_passed++;
    }
} // namespace cuddh_test
