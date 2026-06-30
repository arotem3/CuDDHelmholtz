#pragma once

#include "FEM3D/H1Space3D.hpp"
#include "HostDeviceArray.hpp"
#include "Operator.hpp"
#include "SmallMatrix.hpp"
#include "SparseMatrix.hpp"

namespace cuddh
{
    /// @brief b(u, v) = (grad u, grad v) in 3D.
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

        /// @brief S <- S + c * K  (real sparse matrix)
        bool assemble(double c, SparseMatrix<double> &S_out) const override;

        /// @brief S <- S + c * K  (complex sparse matrix; c may be complex)
        bool assemble(std::complex<double> c, SparseMatrix<double, true> &S_out) const override;

    private:
        const H1Space3D &fem;
        host_device_dvec _D;           // differentiation matrix
        HostDeviceArray<double3x3> _G; // geometric factors
    };
} // namespace cuddh
