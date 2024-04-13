#include "Basis.hpp"

static void barycentric_weights(cuddh::const_dvec_wrapper x, cuddh::dvec& w)
{
    const int n = x.size();

    for (int i=0; i < n; ++i)
    {
        w[i] = 1.0;
        for (int j=0; j < n; ++j)
        {
            if (i == j)
                continue;
            
            w[i] *= x[i] - x[j];
        }
        w[i] = 1.0 / w[i];
    }

    auto [wmin, wmax] = std::minmax_element(w.begin(), w.end());
    double diff = (*wmax) - (*wmin);
    for (int i=0; i < n; ++i)
        w[i] /= diff;
}

/// @brief evaluate lagrange interpolating polynomial at x0
/// @param x0 evaluation point
/// @param x collocation nodes
/// @param w barycentric weights
/// @param y nodal values of interpolant at x
/// @return interpolant evaluated at x0
static double lagrange_interpolation(double x0, cuddh::const_dvec_wrapper x, const cuddh::dvec& w, const cuddh::dvec& y)
{
    const int n = x.size();
    double A = 0.0;
    double B = 0.0;

    constexpr double eps = std::numeric_limits<double>::epsilon();

    for (int i=0; i < n; ++i)
    {
        double xdiff = x0 - x[i];

        if (x0 == x[i] || std::abs(xdiff) <= eps)
            return y[i];

        double C = w[i] / xdiff;
        A += C * y[i];
        B += C;
    }

    return A / B;
}

/// @brief evaluate derivative of lagrange interpolating polynomial at x0
/// @param x0 evaluation point
/// @param x collocation nodes
/// @param w barycentric weights
/// @param y nodal values of interpolant at x
/// @return derivative of interpolant at x0
static double lagrange_derivative(double x0, cuddh::const_dvec_wrapper x, const cuddh::dvec& w, const cuddh::dvec& y)
{
    const int n = x.size();
    bool atnode = false;
    int i;
    
    double A = 0.0;
    double B = 0.0;

    double p = lagrange_interpolation(x0, x, w, y);

    constexpr double eps = std::numeric_limits<double>::epsilon();

    for (int j=0; j < n; ++j)
    {
        if (x0 == x[j] || std::abs(x0 - x[j]) <= eps)
        {
            atnode = true;
            B = -w[j];
            i = j;
        }
    }

    if (atnode)
    {
        for (int j=0; j < n; ++j)
        {
            if (j == i)
                continue;
            
            A += w[j] * (p - y[j]) / (x0 - x[j]);
        }
    }
    else
    {
        for (int j=0; j < n; ++j)
        {
            double t = w[j] / (x0 - x[j]);
            A += t * (p - y[j]) / (x0 - x[j]);
            B += t;
        }
    }

    return A / B;
}

namespace cuddh
{
    Basis::Basis(int n_)
        : n{n_},
          q(n, QuadratureRule::GaussLobatto),
          M(n, n),
          D(n, n),
          wb(n)
    {
        barycentric_weights(q.x(), wb);

        QuadratureRule quad(n, QuadratureRule::GaussLegendre); // need the higher order GaussLegendre rule for mass matrix

        dmat P(n, n);
        eval(n, quad.x(), P);
        
        for (int i=0; i < n; ++i)
        {
            for (int j = 0; j <= i; ++j)
            {
                double m = 0.0;
                for (int k = 0; k < n; ++k)
                {
                    m += quad.w(k) * P(k, i) * P(k, j);
                }
                M(i, j) = m;
                M(j, i) = m;
            }
        }

        deriv(n, q.x(), D);
    }

    void Basis::eval(int m, const double * x, double * P_) const
    {
        auto P = reshape(P_, m, n);
        
        dvec y(n);
        for (int i = 0; i < n; ++i)
        {
            y(i) = 1.0;
            for (int j = 0; j < m; ++j)
            {
                P(j, i) = lagrange_interpolation(x[j], q.x(), wb, y);
            }
            y(i) = 0.0;
        }
    }

    void Basis::deriv(int m, const double * x, double * D_) const
    {
        auto D = reshape(D_, m, n);

        dvec y(n);
        for (int i = 0; i < n; ++i)
        {
            y(i) = 1.0;
            for (int j = 0; j < m; ++j)
            {
                D(j, i) = lagrange_derivative(x[j], q.x(), wb, y);
            }
            y(i) = 0.0;
        }
    }
} // namespace cuddh
