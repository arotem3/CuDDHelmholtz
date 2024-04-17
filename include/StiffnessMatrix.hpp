#ifndef CUDDH_STIFFNESS_MATRIX_HPP
#define CUDDH_STIFFNESS_MATRIX_HPP

#include "Mesh2D.hpp"
#include "Basis.hpp"
#include "H1Space.hpp"

namespace cuddh
{
    /// @brief b(u, v) = (grad u, grad v).
    class StiffnessMatrix
    {
    public:
        StiffnessMatrix(const H1Space& fem);
        StiffnessMatrix(const H1Space& fem, const QuadratureRule& quad);

        /// @brief y <- y + c * S * x, where S is the stiffness matrix. 
        void action(double c, const double * x, double * y) const;

    private:
        const int n_elem;
        const int n_basis;
        const int n_quad;

        const QuadratureRule quad;

        dmat P;
        dmat D;

        TensorWrapper<5, const double> J;
        const_icube_wrapper I;
    };
} // namespace cuddh

#endif