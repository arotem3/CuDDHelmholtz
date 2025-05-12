#ifndef CUDDH_WAVE_EQUATION_HPP
#define CUDDH_WAVE_EQUATION_HPP

#include "cuddh_config.hpp"
#include "cuddh_error.hpp"
#include "Operators2D/StiffnessMatrix.hpp"
#include "Operators2D/MassMatrix.hpp"
#include "Operators2D/FaceMassMatrix.hpp"

namespace cuddh
{
    /**
     * @brief FEM discretization of the wave equation: alpha^2(x) * u_{tt} - div(grad u) == f(x, t)
     * with boundary conditions: alpha(x) * u_t + du/dn == 0.
     */
    class WaveEquation
    {
    public:
        WaveEquation(const double *a2x, const double *ax, const H1Space2D &fem_, const TraceSpace2D &fs_);

        /// @brief u1 ~ u(t + dt) with u0 ~ u(t)
        /// @param dt time step
        /// @param d_u0 DEVICE. current solution u0 = [u, u_t]
        /// @param d_u1 DEVICE. update u1 = [u, u_t]
        /// @param d_acc0 DEVICE. acceleration vector from previous step.
        /// @param d_acc1 DEVICE. updated acceleration vector for next step. On entry, d_acc1 = M * f(x, t + dt)
        void step(double dt, const double *d_u0, double *d_u1, const double *d_acc0, double *d_acc1) const;

        /// @brief compute the acceleration vector for t = t0
        /// @param d_u0 DEVICE. current solution u0 = [u, u_t] at t = t0
        /// @param d_acc0 on entry, d_acc0 = M * f(x, t0). On exit, d_acc0 is set to the acceleration at t = t0.
        void initialize_acceleration(const double *d_u0, double *d_acc0) const;

    private:
        const int ndof; // number of degrees of freedom

        const H1Space2D &fem;
        const TraceSpace2D &fs;

        StiffnessMatrix S;

        HostDeviceArray<double> M; // mass matrix
        HostDeviceArray<double> H; // face mass matrix
    };
} // namespace cuddh

#endif
