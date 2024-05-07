#ifndef CUDDH_QUADRATURE_HPP
#define CUDDH_QUADRATURE_HPP

#include <unordered_map>
#include <cmath>
#include <sstream>
#include <iomanip>

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
        QuadratureRule(const QuadratureRule&) = default;
        QuadratureRule(QuadratureRule&&) = default;
        QuadratureRule& operator=(const QuadratureRule&) = default;
        QuadratureRule& operator=(QuadratureRule&&) = default;

        /// @brief initialize a quadrature rule with n points of type Guass-Lobatto or Gauss-Legendre
        QuadratureRule(int n, QuadratureType type=GaussLobatto);

        /// @brief returns the number of quadrature (point, weight) pairs 
        int size() const
        {
            return _n;
        }

        /// @brief identifies the type of quadrature rule as either Gauss-Legendre or Gauss-Lobatto
        QuadratureType type() const
        {
            return _type;
        }

        /// @brief identifies the quadrature rule by a name of the format "%s%05d" where s is type ("legendre" or "lobatto"), and d is n. 
        std::string name() const;

        /// @brief returns the quadrature points 
        const_dvec_wrapper x() const
        {
            return const_dvec_wrapper(_x.data(), _n);
        }

        /// @brief return the i-th quadrature point 
        double x(int i) const
        {
            return _x(i);
        }

        /// @brief returns the quadrature weights 
        const_dvec_wrapper w() const
        {
            return const_dvec_wrapper(_w.data(), _n);
        }

        /// @brief return the i-th quadrature weight 
        double w(int i) const
        {
            return _w(i);
        }

    private:
        int _n;
        QuadratureType _type;
        dvec _x;
        dvec _w;
    };
} // namespace cuddh

#endif
