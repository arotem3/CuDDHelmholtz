#pragma once

#include "gmres.hpp"

namespace cuddh
{
    solver_out minres(int n, double *x, const Operator *A, const double *b, int maxit = 100, double tol = 1e-6,
                      int verbose = 0);
} // namespace cuddh
