#ifndef CUDDH_STIFFNESS_MATRIX_HPP
#define CUDDH_STIFFNESS_MATRIX_HPP

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "H1Space2D.hpp"
#include "Operator.hpp"
#include "SmallMatrix.hpp"
#include "linalg.hpp"

namespace cuddh
{
    /// @brief b(u, v) = (grad u, grad v).
    class StiffnessMatrix : public Operator<double>
    {
    public:
        explicit StiffnessMatrix(const H1Space2D &fem);

        ~StiffnessMatrix() = default;

        /// @brief y[i] <- y[i] + c * (grad x, grad phi[i])
        /// where phi[i] is the i-th basis function in the H1Space2D
        void action(double c, const double *x, double *y) const override;

        /// @brief y[i] <- (grad x, grad phi[i])
        /// where phi[i] is the i-th basis function in the H1Space2D
        void action(const double *x, double *y) const override;

    private:
        const H1Space2D &fem;

        const int ndof;
        const int n_elem;
        const int n_basis;

        host_device_dvec _D;
        HostDeviceArray<dsym2x2> _G;
    };
} // namespace cuddh

#endif