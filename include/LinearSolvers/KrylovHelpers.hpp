#pragma once
#include <cblas.h>
#include <lapacke.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <algorithm>
#include <limits>
#include <numeric>
#include <vector>

#include "Tensor.hpp"
#include "cuddh_error.hpp"
#include "forall.hpp"
#include "linalg.hpp"

namespace cuddh
{
    template <typename real_t>
    constexpr void apply_givens(real_t &x, real_t &y, real_t c, real_t s)
    {
        real_t t = c * x + s * y;
        y = -s * x + c * y;
        x = t;
    }

    template <typename real_t>
    constexpr std::pair<real_t, real_t> compute_givens(real_t x, real_t y)
    {
        real_t r = std::hypot(x, y);
        if (r == 0)
            return {real_t(1), real_t(0)};
        return {x / r, y / r};
    }

    /**
     * @brief Solves an upper triangular system R * x = b
     *
     * @param ldr Leading dimension of R
     * @param n Size of the system
     * @param R Upper triangular matrix
     * @param b Right-hand side vector, overwritten with the solution
     */
    inline void solve_triu(int ldr, int n, const double *R, double *b)
    {
        int info = LAPACKE_dtrtrs(LAPACK_COL_MAJOR, 'U', 'N', 'N', n, 1, R, ldr, b, n);
        cuddh_verify(info == 0, {
            if (info < 0)
                printf("LAPACKE_dtrtrs: invalid argument index %d\n", -info);
            else
                printf("LAPACKE_dtrtrs: singular matrix; zero diagonal at index %d\n", info);
        });
    }

    /**
     * @brief Solves an upper triangular system R * x = b
     *
     * @param ldr Leading dimension of R
     * @param n Size of the system
     * @param R Upper triangular matrix
     * @param b Right-hand side vector, overwritten with the solution
     */
    inline void solve_triu(int ldr, int n, const float *R, float *b)
    {
        int info = LAPACKE_strtrs(LAPACK_COL_MAJOR, 'U', 'N', 'N', n, 1, R, ldr, b, n);
        cuddh_verify(info == 0, {
            if (info < 0)
                printf("LAPACKE_strtrs: invalid argument index %d\n", -info);
            else
                printf("LAPACKE_strtrs: singular matrix; zero diagonal at index %d\n", info);
        });
    }
} // namespace cuddh
