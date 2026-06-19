#pragma once

#include <cuda_runtime.h>

#include "cuddh_config.hpp"
#include "cuddh_error.hpp"

#ifndef CUDDH_FORALL_BLOCK_SIZE
#define CUDDH_FORALL_BLOCK_SIZE 512
#endif

#define CUDDH_CUDA_CHECK(call)                                                                          \
    do                                                                                                  \
    {                                                                                                   \
        cudaError_t err = (call);                                                                       \
        cuddh_verify(err == cudaSuccess, fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err))); \
    } while (0)

#ifndef NDEBUG
#define CUDDH_CHECK_KERNEL()                       \
    do                                             \
    {                                              \
        CUDDH_CUDA_CHECK(cudaGetLastError());      \
        CUDDH_CUDA_CHECK(cudaDeviceSynchronize()); \
    } while (0)
#else
#define CUDDH_CHECK_KERNEL()                  \
    do                                        \
    {                                         \
        CUDDH_CUDA_CHECK(cudaGetLastError()); \
    } while (0)
#endif

namespace cuddh
{
    template <typename LAMBDA>
    __global__ static void forall_kernel(int n, LAMBDA fun)
    {
        const int k = threadIdx.x + blockIdx.x * blockDim.x;
        if (k >= n)
            return;

        fun(k);
    }

    template <typename LAMBDA>
    inline void forall(int n, LAMBDA &&fun)
    {
        if (n == 0)
            return;

        const int n_blocks = (n + CUDDH_FORALL_BLOCK_SIZE - 1) / CUDDH_FORALL_BLOCK_SIZE;

        forall_kernel<<<n_blocks, CUDDH_FORALL_BLOCK_SIZE>>>(n, fun);
        CUDDH_CHECK_KERNEL();
    }

    template <typename LAMBDA>
    __global__ static void forall_block_kernel(int n, LAMBDA fun)
    {
        const int k = blockIdx.x;
        if (k >= n)
            return;

        fun(k);
    }

    template <typename LAMBDA>
    inline void forall_1d(int bx, int n, LAMBDA &&fun)
    {
        if (n == 0)
            return;

        forall_block_kernel<<<n, bx>>>(n, fun);
        CUDDH_CHECK_KERNEL();
    }

    template <typename LAMBDA>
    inline void forall_2d(int bx, int by, int n, LAMBDA &&fun)
    {
        if (n == 0)
            return;

        const dim3 block_size(bx, by);
        forall_block_kernel<<<n, block_size>>>(n, fun);
        CUDDH_CHECK_KERNEL();
    }

    template <typename LAMBDA>
    inline void forall_3d(int bx, int by, int bz, int n, LAMBDA &&fun)
    {
        if (n == 0)
            return;

        const dim3 block_size(bx, by, bz);
        forall_block_kernel<<<n, block_size>>>(n, fun);
        CUDDH_CHECK_KERNEL();
    }
} // namespace cuddh
