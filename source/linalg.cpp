#include "linalg.hpp"

namespace cuddh
{
    void axpby(int n, double a, const double * x, double b, double * y)
    {
        for (int i=0; i < n; ++i)
            y[i] = a * x[i] + b * y[i];
    }

    double norm(int n, const double * x)
    {
        double d = dot(n, x, x);
        return std::sqrt(d);
    }

    double dot(int n, const double * x, const double * y)
    {
        double s = 0.0;
        for (int i = 0; i < n; ++i)
            s += x[i] * y[i];
        return s;
    }

    void copy(int n, const double * x, double * y)
    {
        for (int i = 0; i < n; ++i)
            y[i] = x[i];
    }

    void scal(int n, double a, double * x)
    {
        for (int i = 0; i < n; ++i)
            x[i] *= a;
    }
} // namespace cuddh
