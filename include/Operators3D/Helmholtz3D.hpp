#pragma once

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
        Helmholtz3D(double omega_, const double *a2x, const double *ax, const H1Space3D &fem, const TraceSpace3D &tr);

        /// @brief y[i] = a(x, phi[i]) where a(u,v) = (grad u, grad v) - omega^2 (u, v) - i*omega <u, v>
        /// @param x the real and imaginary part of the solution
        /// @param y on exit y[i] <- a(x, v[i]) for each v
        void action(const double *x, double *y) const override;

        /// @brief y[i] <- y[i] + c * a(x, phi[i]) where a(u,v) = (grad u, grad v) - omega^2 (u, v) - i*omega <u, v>
        void action(double c, const double *x, double *y) const override
        {
            cuddh_verify(false, printf("Helmholtz3D::action(c, x, y) not implemented\n"));
        }

    private:
        const double omega;
        const int ndof;

        StiffnessMatrix3D S;
        MassMatrix3D M;
        host_device_dvec H;
    };
} // namespace cuddh
