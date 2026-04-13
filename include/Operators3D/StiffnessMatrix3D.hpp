#pragma once

#include "FEM3D/H1Space3D.hpp"
#include "HostDeviceArray.hpp"
#include "Operator.hpp"
#include "SmallMatrix.hpp"

namespace cuddh
{
    class StiffnessMatrix3D : public Operator<double>
    {
    public:
        StiffnessMatrix3D(const H1Space3D &fem);

        ~StiffnessMatrix3D() = default;

        /// @brief y[i] <- y[i] + c * (grad x, grad phi[i])
        /// where phi[i] is the i-th basis function in the H1Space
        void action(double c, const double *x, double *y) const override;

        /// @brief y[i] <- (grad x, grad phi[i])
        /// where phi[i] is the i-th basis function in the H1Space
        void action(const double *x, double *y) const override;

    private:
        const H1Space3D &fem;
        host_device_dvec _D;           // differentiation matrix
        HostDeviceArray<double3x3> _G; // geometric factors
    };
} // namespace cuddh
