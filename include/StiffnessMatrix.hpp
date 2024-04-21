#ifndef CUDDH_STIFFNESS_MATRIX_HPP
#define CUDDH_STIFFNESS_MATRIX_HPP

#include "Operator.hpp"
#include "H1Space.hpp"

namespace cuddh
{
    /// @brief b(u, v) = (grad u, grad v).
    class StiffnessMatrix : public Operator
    {
    public:
        StiffnessMatrix(const H1Space& fem);
        StiffnessMatrix(const H1Space& fem, const QuadratureRule& quad);

        ~StiffnessMatrix() = default;

        /// @brief y[i] <- y[i] + c * (grad x, grad phi[i])
        /// where phi[i] is the i-th basis function in the H1Space
        void action(double c, const double * x, double * y) const override;

        /// @brief y[i] <- (grad x, grad phi[i])
        /// where phi[i] is the i-th basis function in the H1Space
        void action(const double * x, double * y) const override;

    private:
        const int ndof;
        const int n_elem;
        const int n_basis;
        const int n_quad;

        dmat P;
        dmat D;

        Tensor<4, double> G;
        const_icube_wrapper I;
    };
} // namespace cuddh

#endif