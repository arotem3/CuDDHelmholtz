#pragma once

#include <thrust/universal_vector.h>

#include "Operators2D/FaceMassMatrix.hpp"
#include "Operators2D/MassMatrix.hpp"
#include "Operators2D/StiffnessMatrix.hpp"

namespace cuddh
{
    /// @brief FEM discretization of the Helmholtz equation -div(grad u) - omega^2 u == f
    /// with boundary conditions: du/dn + i omega u == 0.
    class Helmholtz : public Operator<double>
    {
    public:
        Helmholtz(double omega_, const double *a2x, const double *ax, const H1Space2D &fem, const TraceSpace2D &fs);

        /// @brief y[i] = a(x, phi[i]) where a(u,v) = (grad u, grad v) - omega^2 (u, v) - i*omega <u, v>
        /// @param x the real and imaginary part of the solution
        /// @param y on exit y[i] <- a(x, v[i]) for each v
        void action(const double *x, double *y) const;

        /// @brief y[i] <- y[i] + c * a(x, phi[i]) where a(u,v) = (grad u, grad v) - omega^2 (u, v) - i*omega <u, v>
        /// @param c
        /// @param x the real and imaginary part of the solution
        /// @param y on exit y[i] <- a(x, v[i]) for each v
        void action(double c, const double *x, double *y) const
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
