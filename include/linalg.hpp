#ifndef CUDDH_LINALG_HPP
#define CUDDH_LINALG_HPP

#include <cmath>

namespace cuddh
{
    /// @brief y <- a * x + b * y
    void axpby(int n, double a, const double * x, double b, double * y);

    /// @brief the euclidean norm of x 
    double norm(int n, const double * x);

    /// @brief the dot product between x and y 
    double dot(int n, const double * x, const double * y);

    /// @brief copy x into y 
    void copy(int n, const double * x, double * y);

    /// @brief x <- a * x 
    void scal(int n, double a, double * x);
} // namespace cuddh

#endif
