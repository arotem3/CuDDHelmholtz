#include "include/QuadratureRule.hpp"

template <typename Map, typename Key>
static bool contains(const Map & map, Key key)
{
    return map.find(key) != map.end();
}

// extern to lapack routine dsteqr for eigevalue decomposition of symmetric
// tridiagonal matrix:
// COMPZ: is a single character, set to 'N' for only eigenvalues
// N: order of matrix
// D: pointer to matrix diagonal (length n)
// E: pointer to matrix off diagonal (length n-1)
// Z: pointer to orthogonal matrix, not necessary here
// LDZ_dummy: leading dim of Z, just set to 1 since it wont be used
// WORK: pointer to array, not needed for COMPZ=='N'
// INFO: 0 on success, <0 failed, >0 couldn't find all eigs
extern "C" int dsteqr_(char* COMPZ, int* N, double* D, double* E, double* Z, int* LDZ_dummy, double* WORK, int* INFO);

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

static inline double square(double x)
{
    return x * x;
}

static void gauss_legendre(int n, double * x, double * w)
{
    if (n < 1)
    {
        std::string msg = "QuadratureRule error: Guass-Legendre rules require n >= 1, but n = " + std::to_string(n) + ".";
        cuddh::cuddh_error(msg.c_str());
    }
    
    static const std::unordered_map<int, std::vector<double>> cached_nodes = {
        {1, {0.0}},
        {2, {-0.577350269189625764509149, 0.577350269189625764509149}},
        {3, {-0.774596669241483377035853, 0.0, 0.774596669241483377035853}},
        {4, {-0.861136311594052575223946, -0.339981043584856264802666, 0.339981043584856264802666, 0.861136311594052575223946}},
        {5, {-0.906179845938663992797627, -0.538469310105683091036314, 0.0, 0.538469310105683091036314, 0.906179845938663992797627}},
        {6, {-0.932469514203152027812302, -0.661209386466264513661400, -0.238619186083196908630502, 0.238619186083196908630502, 0.661209386466264513661400, 0.932469514203152027812302}},
        {7, {-0.949107912342758524526190, -0.741531185599394439863865, -0.405845151377397166906606, 0.0, 0.405845151377397166906606, 0.741531185599394439863865, 0.949107912342758524526190}},
        {8, {-0.960289856497536231683561, -0.796666477413626739591554, -0.525532409916328985817739, -0.183434642495649804939476, 0.183434642495649804939476, 0.525532409916328985817739, 0.796666477413626739591554, 0.960289856497536231683561}},
        {9, {-0.968160239507626089835576, -0.836031107326635794299430, -0.613371432700590397308702, -0.324253423403808929038538, 0.0, 0.324253423403808929038538, 0.613371432700590397308702, 0.836031107326635794299430, 0.968160239507626089835576}},
        {10, {-0.973906528517171720077964, -0.865063366688984510732097, -0.679409568299024406234327, -0.433395394129247190799266, -0.148874338981631210884826, 0.148874338981631210884826, 0.433395394129247190799266, 0.679409568299024406234327, 0.865063366688984510732097, 0.973906528517171720077964}}
    };

    if (contains(cached_nodes, n))
    {
        auto& xq = cached_nodes.at(n);
        for (int i=0; i < n; ++i)
        {
            x[i] = xq.at(i);
        }
    }
    else
    { // Golub-Welsch algorithm
        std::fill_n(x, n, 0.0);

        cuddh::dvec E(n-1);
        for (int i=0; i < n-1; ++i) // off diagonal for symmetric tridiagonal matrix
        {
            double k = i + 1;
            E[i] = k * std::sqrt(1.0 / (4.0 * k * k - 1.0));
        }

        int info;
        char only_eigvals = 'N';
        int LDZ_dummy = 1;

        dsteqr_(&only_eigvals, &n, x, E, nullptr, &LDZ_dummy, nullptr, &info);

        if (info != 0)
            cuddh::cuddh_error("QuadratureRule error: failed to compute eigenvalues of companion matrix.");

        // refine roots with Newton's method
        for (int i=0; i < n/2; ++i)
        {
            for (int j=0; j < 3; ++j) // only three iterations, roots should be good to start with
            {
                double P = jacobiP(n, 0, 0, x[i]);
                double dP = jacobiP_derivative(1, n, 0, 0, x[i]);
                x[i] -= P/dP;
            }

            x[n-1-i] = -x[i];
        }

        if (n & 1)
            x[n/2] = 0.0;
    }

    for (int i=0; i < n; ++i)
        w[i] = 2.0 / (1.0 - square(x[i])) / square(jacobiP_derivative(1, n, 0, 0, x[i]));
}

static void gauss_lobatto(int n, double * x, double * w)
{
    if (n < 2)
    {
        std::string msg = "QuadratureRule error: Gauss-Lobatto rules require n >= 2, but n =" + std::to_string(n) + ".";
        cuddh::cuddh_error(msg.c_str());
    }
        
    static const std::unordered_map<int, std::vector<double>> cached_nodes = {
        {2, {-1,1}},
        {3, {-1,0,1}},
        {4, {-1, -0.447213595499958, 0.447213595499958, 1}},
        {5, {-1, -0.654653670707977, 0, 0.654653670707977, 1}},
        {6, {-1, -0.765055323929465, -0.285231516480645, 0.285231516480645, 0.765055323929465, 1}},
        {7, {-1, -0.830223896278567, -0.468848793470714, 0.0, 0.468848793470714, 0.830223896278567, 1}},
        {8, {-1, -0.871740148509607, -0.591700181433142, -0.209299217902479, 0.2092992179024789, 0.591700181433142, 0.871740148509607, 1}},
        {9, {-1, -0.899757995411460, -0.677186279510738, -0.363117463826178, 0, 0.363117463826178, 0.677186279510738, 0.899757995411460, 1}}
    };
    
    if (contains(cached_nodes, n))
    {
        auto& xq = cached_nodes.at(n);
        for (int i=0; i < n; ++i)
            x[i] = xq.at(i);
    }
    else
    { // use the Golub-Welsch algorithm
        std::fill_n(x+1, n-2, 0.0);

        cuddh::dvec E(n-3); // off-diagonal of symmetric tridiagonal matrix
        for (int i=0; i < n-3; ++i)
        {
            double ii = i+1;
            E[i] = std::sqrt( ii * (ii + 2.0) / ((2.0*ii + 3.0) * (2.0*ii + 1.0)) );
        }

        int N = n-2;
        int info;
        char only_eigvals = 'N';
        int LDZ_dummy = 1;

        dsteqr_(&only_eigvals, &N, x+1, E, nullptr, &LDZ_dummy, nullptr, &info);

        if (info != 0)
            cuddh::cuddh_error("QuadratureRule error: eigenvalue decomposition failed!");

        x[0] = -1.0;
        x[n-1] = 1.0;

        // refine roots with Newton's method
        for (int i=1; i < n/2; ++i)
        {
            for (int j=0; j < 3; ++j) // just three iterations, shouldn't need much refinement
            {
                double P = jacobiP(n-2, 1, 1, x[i]);
                double dP = jacobiP_derivative(1, n-2, 1, 1, x[i]);
                x[i] -= P/dP;
            }

            x[n-1-i] = -x[i];
        }

        if (n & 1)
            x[n/2] = 0.0;
    }

    for (int i=0; i < n; ++i)
        w[i] = 2.0 / (n*(n-1) * square(jacobiP(n-1, 0, 0, x[i])));
}

namespace cuddh
{
    QuadratureRule::QuadratureRule() : _n{0}, _type{GaussLobatto}, _x(), _w() {}

    QuadratureRule::QuadratureRule(int n, QuadratureType type)
        : _n{n},
          _type{type},
          _x(n),
          _w(n)
    {
        if (type == GaussLegendre)
            gauss_legendre(n, _x, _w);
        else // (type == GaussLobatto)
            gauss_lobatto(n, _x, _w);
    }

    std::string QuadratureRule::name() const
    {
        std::stringstream s;
        if (_type == GaussLegendre)
            s << "legendre";
        else
            s << "lobatto";
        s << std::setw(5) << std::setfill('0') << _n;
        return s.str();
    }
} // namespace cuddh
