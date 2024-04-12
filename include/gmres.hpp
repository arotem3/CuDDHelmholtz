#ifndef CUDDH_GMRES_HPP
#define CUDDH_GMRES_HPP

#include "Tensor.hpp"

namespace cuddh
{
    template <typename Matrix, typename Preconditioner>
    void gmres(int n, double * x, Matrix A, const double * b, Preconditioner M, int m, int maxit, double tol);
} // namespace cuddh


#endif