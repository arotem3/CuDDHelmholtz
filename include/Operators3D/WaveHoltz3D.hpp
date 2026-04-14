#pragma once

#include <cmath>

#include "FEM3D/GridFunc3D.hpp"
#include "HostDeviceArray.hpp"
#include "Operators3D/FaceMassMatrix3D.hpp"
#include "Operators3D/MassMatrix3D.hpp"
#include "Operators3D/StiffnessMatrix3D.hpp"
#include "linalg.hpp"

namespace cuddh
{
    class WaveHoltz3D : public Operator<double>
    {
    public:
        /// @brief Constant-coefficient constructor: a(x) = 1.
        WaveHoltz3D(const H1Space3D &fem, const TraceSpace3D &fs, double omega);

        /// @brief Variable-coefficient constructor.
        /// @param a GridFunc3D representing the wave-speed coefficient a(x).
        WaveHoltz3D(const H1Space3D &fem, const TraceSpace3D &fs, double omega, const GridFunc3D<double> &a);

        /// @brief y <- y + c * S * x
        inline void S(double c, const double *x, double *y) const
        {
            evolve_project(c, x, nullptr, y); // y <- y + c * S(x)
        }

        /// @brief Gf <- G * f
        inline void G(const double *f, double *Gf) const
        {
            dla::zeros(this->ndof(), Gf);        // Gf <- 0
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

        int nt;
        double shift;

        StiffnessMatrix3D stiffness;
        HostDeviceArray<double2> ab;

        mutable HostDeviceArray<double> acc;
        mutable HostDeviceArray<double> w;

        inline double filter(double n) const { return (2.0 / nt) * (std::cos(2.0 * M_PI * n / nt) - shift); }
    };
} // namespace cuddh
