#ifndef CUDDH_WAVEHOLTZ_HPP
#define CUDDH_WAVEHOLTZ_HPP

#include "Operators2D/FaceMassMatrix.hpp"
#include "Operators2D/MassMatrix.hpp"
#include "Operators2D/StiffnessMatrix.hpp"
#include "cuddh_config.hpp"
#include "cuddh_error.hpp"
#include "linalg.hpp"

namespace cuddh
{
    /**
     * @brief WaveHoltz FEM solver for the Helmholtz equation: -div(grad u) - omega^2 alpha^2(x) u = f(x)
     * with boundary conditions: du/dn - i omega alpha(x) u == 0.
     *
     * Can be used in the fixed point iteration:
     *  u[n+1] = S(u[n]) + G(f)
     *
     * Or accelerated with gmres as the system:
     *  action(u) = G(f).
     */
    class WaveHoltz : public Operator
    {
    public:
        WaveHoltz(double omega, double maximum_velocity, const double *a2x, const double *ax, const H1Space2D &fem_,
                  const TraceSpace2D &fs_);

        /// @brief y <- y + c * S * x
        inline void S(double c, const double *x, double *y) const
        {
            evolve_project(c, x, nullptr, y); // y <- y + c * S(x)
        }

        /// @brief Gf <- G * f
        inline void G(const double *f, double *Gf) const
        {
            evolve_project(1.0, nullptr, f, Gf); // Gf <- G * f
        }

        /// @brief y <- y + c * (I - S) * x
        void action(double c, const double *x, double *y) const override;

        /// @brief y <- (I - S) * x
        void action(const double *x, double *y) const override;

        /// @brief out <- out + c * (S * u + G * f)
        void evolve_project(double c, const double *u, const double *f, double *out) const;

    private:
        const double omega; // Helmholtz frequency
        const int ndof;     // number of degrees of freedom

        int nt;
        double shift;

        StiffnessMatrix stiffness;
        MassMatrix mass;
        FaceMassMatrix face_mass;

        mutable HostDeviceArray<double> acc;
        mutable HostDeviceArray<double> w;

        inline double filter(double n) const
        {
            return (2.0 / nt) * (std::cos(2.0 * M_PI * n / nt) - shift);
        }
    };
} // namespace cuddh

#endif
