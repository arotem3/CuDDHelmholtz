#ifndef CUDDH_QUADRATURE_HPP
#define CUDDH_QUADRATURE_HPP

#include <cmath>
#include <iomanip>
#include <sstream>
#include <unordered_map>

#include "HostDeviceArray.hpp"
#include "Tensor.hpp"
#include "cuddh_error.hpp"

namespace cuddh
{
    /// @brief quadrature rule for the interval [-1, 1]
    class QuadratureRule
    {
    public:
        enum QuadratureType
        {
            GaussLegendre,
            GaussLobatto
        };

        QuadratureRule();
        QuadratureRule(const QuadratureRule &) = default;
        QuadratureRule(QuadratureRule &&) = default;
        QuadratureRule &operator=(const QuadratureRule &) = default;
        QuadratureRule &operator=(QuadratureRule &&) = default;

        /// @brief initialize a quadrature rule with n points of type Guass-Lobatto or Gauss-Legendre
        QuadratureRule(int n, QuadratureType type = GaussLobatto);

        /// @brief returns the number of quadrature (point, weight) pairs
        int size() const { return _n; }

        /// @brief identifies the type of quadrature rule as either Gauss-Legendre or Gauss-Lobatto
        QuadratureType type() const { return _type; }

        /// @brief identifies the quadrature rule by a name of the format "%s%05d" where s is type ("legendre" or
        /// "lobatto"), and d is n.
        std::string name() const;

        /// @brief returns the quadrature points
        const_dvec_wrapper x(MemorySpace ms) const { return reshape(_x.read(ms), _n); }

        /// @brief returns the quadrature weights
        const_dvec_wrapper w(MemorySpace ms) const { return reshape(_w.read(ms), _n); }

    private:
        int _n;
        QuadratureType _type;
        HostDeviceArray<double> _x;
        HostDeviceArray<double> _w;
    };
} // namespace cuddh

#endif
