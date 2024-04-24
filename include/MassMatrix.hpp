#ifndef CUDDH_MASS_MATRIX_HPP
#define CUDDH_MASS_MATRIX_HPP

#include "Basis.hpp"
#include "Mesh2D.hpp"
#include "H1Space.hpp"

#include "HostDeviceArray.hpp"
#include "forall.hpp"
#include "linalg.hpp"

namespace cuddh
{
    /// @brief m(u, v) = (u, v) or m(u, v) = (a(x)*u, v)
    class MassMatrix : public Operator
    {
    public:
        /// @brief initialize mass matrix m(u, v) = (u, v)
        MassMatrix(const H1Space& fem);

        /// @brief initialize weighted mass matrix m(u, v) = (a(x)*u, v)
        /// @param a DEVICE. H1Space vector representing the function a.
        /// @param fem 
        MassMatrix(const double * a, const H1Space& fem);

        /// @brief y <- y + c * M*x, where M is the mass matrix
        void action(double c, const double * x, double * y) const override;

        void action(const double * x, double * y) const override;

    private:
        const H1Space& fem;
        
        const int ndof;
        const int n_elem;
        const int n_basis;
        const int n_quad;
        const QuadratureRule quad;
        
        host_device_dvec _P;
        host_device_dvec _a; // a(x) * w(i) * w(j) * detJ
    };

    /// @brief diagonal approximate inverse of mass matrix
    class DiagInvMassMatrix : public Operator
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
        void action(double c, const double * x, double * y) const override;

        void action(const double * x, double * y) const override;

    private:
        const H1Space& fem;
        const int ndof;

        host_device_dvec _p;
    };
} // namespace cuddh


#endif