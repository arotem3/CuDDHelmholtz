#ifndef CUDDH_WAVEHOLTZ3D_HPP
#define CUDDH_WAVEHOLTZ3D_HPP

#include "cuddh_config.hpp"
#include "cuddh_error.hpp"
#include "Operators3D/StiffnessMatrix3D.hpp"
#include "Operators3D/MassMatrix3D.hpp"
#include "Operators3D/FaceMassMatrix3D.hpp"

namespace cuddh
{
    class WaveHoltz3D : public Operator
    {
    public:
        WaveHoltz3D(double omega, double maximum_velocity, const double * a2x, const double * ax, const H1Space3D& fem_, const TraceSpace3D& fs_);

        /// @brief y <- y + c * S * x
        inline void S(double c, const double * x, double * y) const
        {
            evolve_project(c, x, nullptr, y); // y <- y + c * S(x)
        }

        /// @brief Gf <- G * f
        inline void G(const double * f, double * Gf) const
        {
            zeros(2*ndof, Gf); // Gf <- 0
            evolve_project(1.0, nullptr, f, Gf); // Gf <- G * f
        }

        /// @brief y <- y + c * (I - S) * x 
        void action(double c, const double * x, double * y) const override;
        
        /// @brief y <- (I - S) * x
        void action(const double * x, double * y) const override;

        /// @brief out <- out + c * (S * u + G * f)
        void evolve_project(double c, const double * u, const double * f, double * out) const;

    private:
        const double omega; // Helmholtz frequency
        const int ndof; // number of degrees of freedom

        int nt;
        double dt;
        double theta;
        double sigma;
        double shift;

        const H1Space3D& fem;
        const TraceSpace3D& fs;

        StiffnessMatrix3D stiffness;
        MassMatrix3D M;
        HostDeviceArray<double> H; // face mass matrix

        mutable HostDeviceArray<double> acc;
        mutable HostDeviceArray<double> acc1;
        mutable HostDeviceArray<double> w;

        inline double filter(double t) const
        {
            return (2.0 / nt) * (std::cos(omega * t) - shift);
        }
    };
} // namespace cuddh

#endif
