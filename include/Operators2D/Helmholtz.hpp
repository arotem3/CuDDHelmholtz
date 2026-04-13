#pragma once

#include "FEM2D/GridFunc2D.hpp"
#include "FEM2D/H1Space2D.hpp"
#include "Operators2D/FaceMassMatrix.hpp"
#include "Operators2D/MassMatrix.hpp"
#include "Operators2D/StiffnessMatrix.hpp"
#include "cuddh_error.hpp"

namespace cuddh
{
    /// @brief FEM discretization of the Helmholtz equation -div(grad u) - a(x)^2 omega^2 u == f
    /// with boundary conditions: du/dn - i omega a(x) u == 0.
    class Helmholtz : public Operator<double>
    {
    public:
        Helmholtz(const H1Space2D &fem, const TraceSpace2D &fs, double omega);
        Helmholtz(const H1Space2D &fem, const TraceSpace2D &fs, double omega, const GridFunc2D<double> &a);

        /// @brief y[i] = a(x, phi[i]) where a(u,v) = (grad u, grad v) - omega^2 (u, v) - i*omega <u, v>
        /// @param x the real and imaginary part of the solution
        /// @param y on exit y[i] <- a(x, v[i]) for each v
        void action(const double *x, double *y) const;

        void action(double, const double *, double *) const
        {
            cuddh_verify(false, printf("Helmholtz::action(c, x, y) not implemented\n"));
        }

    private:
        const double omega;

        StiffnessMatrix S;
        MassMatrix M;
        FaceMassMatrix H;
    };
} // namespace cuddh
