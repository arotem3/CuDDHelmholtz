#ifndef CUDDH_WAVEHOLTZ3D_HPP
#define CUDDH_WAVEHOLTZ3D_HPP

#include "Operators3D/FaceMassMatrix3D.hpp"
#include "Operators3D/MassMatrix3D.hpp"
#include "Operators3D/StiffnessMatrix3D.hpp"
#include "cuddh_config.hpp"
#include "cuddh_error.hpp"

namespace cuddh
{
    class WaveHoltz3D : public Operator<double>
    {
    public:
        WaveHoltz3D(double omega, const double *a2x, const double *ax, const H1Space3D &fem_, const TraceSpace3D &fs_);

        /// @brief y <- y + c * S * x
        inline void S(double c, const double *x, double *y) const
        {
            evolve_project(c, x, nullptr, y); // y <- y + c * S(x)
        }

        /// @brief Gf <- G * f
        inline void G(const double *f, double *Gf) const
        {
            dla::zeros(2 * ndof, Gf);            // Gf <- 0
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

        StiffnessMatrix3D stiffness;
        HostDeviceArray<double2> ab; // time stepping

        mutable HostDeviceArray<double> acc;
        mutable HostDeviceArray<double> w;

        inline double filter(double n) const { return (2.0 / nt) * (std::cos(2.0 * M_PI * n / nt) - shift); }
    };
} // namespace cuddh

#endif
