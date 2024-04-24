#ifndef CUDDH_LINALG_HPP
#define CUDDH_LINALG_HPP

#include <cmath>
#include <assert.h>

#include <cuda_runtime.h>

#include "cuddh_config.hpp"
#include "forall.hpp"
#include "HostDeviceArray.hpp"

namespace cuddh
{
    /// @brief y <- a * x + b * y
    void axpby(int n, double a, const double * x, double b, double * y);

    /// @brief the euclidean norm of x 
    double norm(int n, const double * x);

    /// @brief the dot product between x and y 
    double dot(int n, const double * x, const double * y);

    /// @brief ||x - y||
    double dist(int n, const double * x, const double * y);

    /// @brief copy x into y 
    void copy(int n, const double * x, double * y);

    /// @brief x <- a * x 
    void scal(int n, double a, double * x);

    /// @brief x <- a
    void fill(int n, double a, double * x);

    /// @brief x <- 0
    void zeros(int n, double * x);

    /// @brief x <- 1
    void ones(int n, double * x);
} // namespace cuddh

#endif
