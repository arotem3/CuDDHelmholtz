#pragma once

#include "FEM3D/GridFunc3D.hpp"
#include "Operators3D/FaceMassMatrix3D.hpp"
#include "Operators3D/MassMatrix3D.hpp"
#include "Operators3D/StiffnessMatrix3D.hpp"
#include "cuddh_error.hpp"

namespace cuddh
{
    /// @brief FEM discretization of the 3D Helmholtz equation -div(grad u) - omega^2 u == f
    /// with boundary conditions: du/dn + i omega u == 0.
    class Helmholtz3D : public Operator<double>
    {
    public:
        /// @brief Constant-coefficient constructor: a(x) = 1.
        Helmholtz3D(const H1Space3D &fem, const TraceSpace3D &tr, double omega);

        /// @brief Variable-coefficient constructor.
        /// @param a GridFunc3D representing the wave-speed coefficient a(x).
        ///          Internally squares a for the mass matrix and computes trace(tr, a)
        ///          for the face mass term.
        Helmholtz3D(const H1Space3D &fem, const TraceSpace3D &tr, double omega, const GridFunc3D<double> &a);

        /// @brief y[i] = a(x, phi[i]) where a(u,v) = (grad u, grad v) - omega^2 (u, v) - i*omega <u, v>
        /// @param x the real and imaginary part of the solution
        /// @param y on exit y[i] <- a(x, v[i]) for each v
        void action(const double *x, double *y) const override;

        void action(double, const double *, double *) const override
        {
            cuddh_verify(false, printf("Helmholtz3D::action(c, x, y) not implemented\n"));
        }

    private:
        const double omega;

        StiffnessMatrix3D S;
        MassMatrix3D M;
        FaceMassMatrix3D H;
    };
} // namespace cuddh
