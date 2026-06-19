#pragma once

#include "FEM3D/GridFunc3D.hpp"
#include "Operators3D/FaceMassMatrix3D.hpp"
#include "Operators3D/MassMatrix3D.hpp"
#include "Operators3D/StiffnessMatrix3D.hpp"

namespace cuddh
{
    /**
     * @brief SEM discretization of the wave equation: alpha^2(x) * u_{tt} - div(grad u) = f(x, t),
     * with boundary conditions: alpha(x) * u_t + du/dn = 0.
     */
    class WaveEquation3D
    {
    public:
        /// @brief Constant-coefficient constructor: a(x) = 1.
        WaveEquation3D(const H1Space3D &fem, const TraceSpace3D &tr);

        /// @brief Variable-coefficient constructor.
        /// @param a GridFunc3D representing the wave-speed coefficient a(x).
        WaveEquation3D(const H1Space3D &fem, const TraceSpace3D &tr, const GridFunc3D<double> &a);

        /**
         * @brief u1 = u(t + dt) with u0 = u(t).
         *
         * @param dt time step
         * @param d_u0 DEVICE. current solution u0 = [u(t), u_t(t)]
         * @param du1 DEVICE. update u1 = [u(t+dt), u_t(t+dt)]
         * @param d_acc0 DEVICE. acceleration vector at t.
         * @param d_acc1 DEVICE. acceleration vector at t+dt. On entry, d_acc1 = M * f(x, t + dt).
         */
        void step(double dt, const double *d_u0, double *du1, const double *d_acc0, double *d_acc1) const;

        /**
         * @brief Compute the acceleration vector at initial t.
         *
         * @param d_u0 DEVICE. current solution u0 = [u(t0), u_t(t0)]
         * @param d_acc0 DEVICE. on entry, d_acc0 = M * f(x, t0). On exit, d_acc0 is set to the acceleration at t = t0.
         */
        void initialize_acceleration(const double *d_u0, double *d_acc0) const;

    private:
        const int ndof;

        const H1Space3D &fem;
        const TraceSpace3D &tr;

        StiffnessMatrix3D S;
        MassMatrix3D M;
        FaceMassMatrix3D H;
    };
} // namespace cuddh
