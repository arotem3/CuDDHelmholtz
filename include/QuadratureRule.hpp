#ifndef CUDDH_QUADRATURE_HPP
#define CUDDH_QUADRATURE_HPP

#include <unordered_map>
#include <cmath>

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

        /// @brief initialize a quadrature rule with n points of type Guass-Lobatto or Gauss-Legendre
        QuadratureRule(int n, QuadratureType type=GaussLobatto);

        /// @brief returns the number of quadrature (point, weight) pairs 
        int size() const
        {
            return _n;
        }

        /// @brief returns the quadrature points 
        const_dvec_wrapper x() const
        {
            return const_dvec_wrapper(_x.data(), _n);
        }

        /// @brief returns the quadrature weights 
        const_dvec_wrapper w() const
        {
            return const_dvec_wrapper(_w.data(), _n);
        }

    private:
        const int _n;
        dvec _x;
        dvec _w;
    };
} // namespace cuddh

#endif
