#pragma once

#include "cuddh_config.hpp"
#include "forall.hpp"

namespace cuddh
{
    // Applies R = 0.5*[1-i, 1+i; 1+i, 1-i] which symmetrizes the DDH operator.
    // If x is provided, y <- R * (x - y), otherwise y <- R * y
    void symmetrize_ddh(int n, const float *x, float *y);
    void symmetrize_ddh(int n, const double *x, double *y);
} // namespace cuddh
