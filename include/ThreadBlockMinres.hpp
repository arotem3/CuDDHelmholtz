#pragma once

#include <cub/cub.cuh>
#include <cuda/std/array>

namespace cuddh
{
    template <typename real_t, int T, int BX, int BY = 1, int BZ = 1>
    struct ThreadBlockMinres
    {
        using BlockReduce = cub::BlockReduce<real_t, BX, cub::BLOCK_REDUCE_WARP_REDUCTIONS, BY, BZ>;
        struct SharedResources
        {
            typename BlockReduce::TempStorage reduce_work;
            real_t reduce_result;
        };

        using arr_t = cuda::std::array<real_t, T>;

        SharedResources &smem;
        arr_t v, w, wp, vp, wpp;

        __device__ ThreadBlockMinres(SharedResources &shared) : smem{shared} {}

        // solves A x == b. On entry r = b and r is overwritten during the solve. On exit, x is the solution and solve
        // returns the norm of the residual.
        template <typename A_t>
        real_t __device__ solve(arr_t &x, A_t &&A, arr_t &r, int max_iter, real_t rtol, real_t atol)
        {
            auto dot = [&](const arr_t &a, const arr_t &b) {
                real_t sum = 0;
                for (int i = 0; i < T; ++i)
                    sum += a[i] * b[i];
                sum = BlockReduce(smem.reduce_work).Sum(sum);
                if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0)
                    smem.reduce_result = sum;
                __syncthreads();

                return smem.reduce_result;
            };

            auto norm = [&](const arr_t &a) {
                return sqrt(dot(a, a));
            };

            wp = arr_t{};
            vp = arr_t{};
            wpp = arr_t{};

            real_t tol = max(rtol * norm(r), atol);

            // r = b - A*x (on input r = b)
            v = A(x);
            for (int t = 0; t < T; ++t)
                r[t] -= v[t];

            real_t phi = norm(r);

            if (phi <= atol)
                return phi;

            real_t recip = 1 / phi;
            for (int t = 0; t < T; ++t)
                v[t] = r[t] * recip;

            real_t cp = 1.0, sp = 0.0;
            real_t c = 1.0, s = 0.0;
            real_t beta = 0.0;

            for (int it = 0; it < max_iter && abs(phi) > tol; ++it)
            {
                r = A(v);
                real_t alpha = dot(r, v);

                for (int t = 0; t < T; ++t)
                    r[t] -= alpha * v[t] + beta * vp[t];

                real_t rho2 = sp * beta;
                real_t gamma = cp * beta;
                real_t rho1 = c * gamma + s * alpha;
                real_t delta = -s * gamma + c * alpha;

                beta = norm(r);

                real_t rho3 = hypot(delta, beta);
                if (rho3 <= 0)
                    break;
                rho3 = 1 / rho3;

                cp = c;
                sp = s;
                c = delta * rho3;
                s = beta * rho3;

                for (int t = 0; t < T; ++t)
                {
                    w[t] = (v[t] - rho1 * wp[t] - rho2 * wpp[t]) * rho3;
                    x[t] += c * phi * w[t];
                }

                phi *= -s;

                for (int t = 0; t < T; ++t)
                {
                    wpp[t] = wp[t];
                    wp[t] = w[t];
                    vp[t] = v[t];
                    v[t] = r[t] / beta;
                }
            }

            return abs(phi);
        }
    };
} // namespace cuddh
