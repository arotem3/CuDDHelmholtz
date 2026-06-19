#pragma once

#include "Basis.hpp"
#include "QuadratureRule.hpp"

namespace cuddh
{
    void polynomial_projection_matrix(MatrixWrapper<double> P, const Basis &basis, const QuadratureRule &quad);
} // namespace cuddh
