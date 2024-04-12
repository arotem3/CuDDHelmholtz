#ifndef CUDDH_STIFFNESS_MATRIX_HPP
#define CUDDH_STIFFNESS_MATRIX_HPP

#include "Mesh2D.hpp"
#include "Basis.hpp"

namespace cuddh
{
    /// @brief b(u, v) = (grad u, grad v).
    class StiffnessMatrix
    {
    public:
        StiffnessMatrix(const Basis& basis, const Mesh2D& mesh);

        /// @brief y <- y + c * S * x, where S is the stiffness matrix. 
        void action(double c, const double * x, double * y) const;

    private:
        const_dmat_wrapper D;
        double J;
    };
} // namespace cuddh


#endif