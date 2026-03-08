#pragma once

#include <thrust/device_vector.h>

#include <chrono>
#include <format>
#include <iomanip>
#include <iostream>

#include "Operator.hpp"
#include "Tensor.hpp"
#include "linalg.hpp"

namespace cuddh
{
    struct SolverResults
    {
        bool success;
        int num_iter;
        int num_matvec;
        std::vector<double> res_norm;
        std::vector<double> time;
    };
} // namespace cuddh
