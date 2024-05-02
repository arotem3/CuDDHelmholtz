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
    /// @brief y[i] <- a * x[i] + b * y[i]
    void axpby(int n, double a, const double * x, double b, double * y);
    void axpby(int n, float a, const float * x, float b, float * y);

    /// @brief the dot product between x and y 
    double dot(int n, const double * x, const double * y);
    float dot(int n, const float * x, const float * y);

    /// @brief the euclidean norm of x: ||x||
    inline double norm(int n, const double * x) {return std::sqrt(dot(n,x,x));}
    inline float norm(int n, const float * x) {return std::sqrt(dot(n,x,x));}

    /// @brief ||x - y||
    double dist(int n, const double * x, const double * y);
    float dist(int n, const float * x, const float * y);

    /// @brief y[i] <- x[i]
    void copy(int n, const double * x, double * y);
    void copy(int n, const float * x, float * y);
    void copy(int n, const int * x, int * y);

    /// @brief x[i] <- a * x[i]
    void scal(int n, double a, double * x);
    void scal(int n, float a, float * x);

    /// @brief x[i] <- a
    void fill(int n, double a, double * x);
    void fill(int n, float a, float * x);
    void fill(int n, int a, int * x);

    /// @brief x[i] <- 0
    inline void zeros(int n, double * x) {fill(n, 0.0, x);}
    inline void zeros(int n, float * x) {fill(n, 0.0f, x);}
    inline void zeros(int n, int * x) {fill(n, 0, x);}

    /// @brief x[i] <- 1
    inline void ones(int n, double * x) {fill(n, 1.0, x);}
    inline void ones(int n, float * x) {fill(n, 1.0f, x);}
    inline void ones(int n, int * x) {fill(n, 1, x);}
} // namespace cuddh

#endif
