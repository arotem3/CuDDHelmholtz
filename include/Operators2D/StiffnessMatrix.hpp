#pragma once

#include "FEM2D/H1Space2D.hpp"
#include "HostDeviceArray.hpp"
#include "Operator.hpp"
#include "SmallMatrix.hpp"
#include "SparseMatrix.hpp"

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

        /// @brief S <- S + c * K  (real sparse matrix)
        bool assemble(double c, SparseMatrix<double> &S_out) const override;

        /// @brief S <- S + c * K  (complex sparse matrix; c may be complex)
        bool assemble(std::complex<double> c, SparseMatrix<double, true> &S_out) const override;

    private:
        const H1Space2D &fem;

        const int n_elem;
        const int n_basis;

        host_device_dvec _D;
        HostDeviceArray<dsym2x2> _G;
    };
} // namespace cuddh
