#ifndef CUDDH_MASS_MATRIX_HPP
#define CUDDH_MASS_MATRIX_HPP

#include "Basis.hpp"
#include "Mesh2D.hpp"
#include "H1Space.hpp"

namespace cuddh
{
    /// @brief m(u, v) = (u, v) or m(u, v) = (a(x)*u, v)
    class MassMatrix
    {
    public:
        /// @brief initialize mass matrix m(u, v) = (u, v)
        MassMatrix(const H1Space& fem);

        /// @brief initialize weighted mass matrix m(u, v) = (a(x)*u, v)
        /// @param a nodal-FEM grid function representing the coefficient a(x)
        /// on the nodes of the mesh
        /// @param fem 
        MassMatrix(const double * a, const H1Space& fem);

        /// @brief y <- y + c * M*x, where M is the mass matrix
        void action(double c, const double * x, double * y) const;

    private:
        const int n_elem;
        const int n_basis;
        const int n_quad;
        const QuadratureRule quad;
        
        dmat P;
        dcube a; // a(x) * w(i) * w(j) * detJ

        const_icube_wrapper I;
    };

    /// @brief diagonal approximate inverse of mass matrix
    class DiagInvMassMatrix
    {
    public:
        /// @brief construct a diagonal approximate inverse of the mass matrix
        /// m(u, v) = (u, v).
        DiagInvMassMatrix(const H1Space& fem);

        /// @brief construct a diagonal approximate inverse of the mass matrix
        /// m(u, v) = (a(x)*u, v).
        /// @param a nodal-FEM grid function representing the coefficient a(x)
        /// on the nodes of the mesh
        /// @param fem 
        DiagInvMassMatrix(const double * a, const H1Space& fem);

        /// @brief y <- y + c * P*x where P ~ inv(M), where M is the mass matrix. 
        void action(double c, const double * x, double * y) const;

    private:
        const int ndof;

        dvec p;
        const_icube_wrapper I;
    };
} // namespace cuddh


#endif