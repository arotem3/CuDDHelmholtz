#ifndef CUDDH_BASIS_HPP
#define CUDDH_BASIS_HPP

#include "Tensor.hpp"
#include "QuadratureRule.hpp"

#include <algorithm>

namespace cuddh
{
    /// @brief 1D nodal basis functions on Gauss-Lobatto nodes.
    class Basis
    {
    public:
        /// @brief initialize a basis set on n Guass-Lobatto nodes.
        Basis(int n);

        /// @brief return the number of basis functions
        int size() const
        {
            return n;
        }

        /// @brief evaluate basis function on grid x
        /// @param n number of points in grid
        /// @param x length n, grid points in [-1, 1] to evaluate basis on
        /// @param P shape (n, n_basis), P(i, j) is the j-th basis function
        /// evaluated at x[i].
        void eval(int n, const double * x, double * P) const;

        /// @brief evaluate derivative of basis function on grid x
        /// @param n number of points in grid
        /// @param x length n, grid points in [-1, 1] to evaluate basis on
        /// @param D shape (n, n_basis), D(i, j) is the derivative of the j-th
        /// basis function evaluated at x[i].
        void deriv(int n, const double * x, double * D) const;

        /// @brief returns the mass matrix for the basis set
        const_dmat_wrapper mass_matrix() const
        {
            return const_dmat_wrapper(M.data(), n, n);
        }

        /// @brief returns the derivative matrix for the basis set
        const_dmat_wrapper derivative_matrix() const
        {
            return const_dmat_wrapper(D.data(), n, n);
        }

        /// @brief returns the underlying quadrature rule on which the nodal
        /// basis is defined.
        const QuadratureRule& quadrature() const
        {
            return q;
        }

    private:
        int n;
        QuadratureRule q;
        dmat M; // mass matrix
        dmat D; // derivative matrix
        dvec wb; // barycentric weights (for interpolation)
    };
} // namespace cuddh


#endif