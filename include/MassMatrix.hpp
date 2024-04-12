#ifndef CUDDH_MASS_MATRIX_HPP
#define CUDDH_MASS_MATRIX_HPP

#include "Basis.hpp"
#include "Mesh2D.hpp"

namespace cuddh
{
    /// @brief m(u, v) = (u, v)
    class MassMatrix
    {
    public:
        MassMatrix(const Basis& basis, const Mesh2D& mesh);

        /// @brief y <- y + a * M*x, where M is the mass matrix
        void action(double a, const double * x, double * y) const;

    private:
        const_dmat_wrapper M;
        double J; // jacobian
    };

    /// @brief a(u, v) = (a(x)*u, v)
    class WeightedMassMatrix
    {
    public:
        /// @brief initialize weighted mass matrix
        /// @param a nodal-FEM grid function representing the coefficient a(x)
        /// on the nodes of the mesh
        /// @param basis 1D basis functions for tensor product
        /// @param mesh the two dimensional mesh
        WeightedMassMatrix(const double * a, const Basis& basis, const Mesh2D& mesh);

        /// @brief y <- y + c * A * x, where A is the weighted mass matrix 
        void action(double c, const double * x, double * y) const;
    
    private:
        const Basis& basis;
        QuadratureRule quad;
        const_dcube_wrapper a;
        double J;
    };
} // namespace cuddh


#endif