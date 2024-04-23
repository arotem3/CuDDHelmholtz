#ifndef CUDDH_FORALL_HPP
#define CUDDH_FORALL_HPP

#include <cuda_runtime.h>

namespace cuddh
{
    template <typename LAMBDA>
    __global__ static void forall_kernel(int n, LAMBDA fun)
    {
        const int k = threadIdx.x + blockIdx.x * blockDim.x;
        if (k >= n) return;
        
        fun(k);
    }

    template <typename LAMBDA>
    void forall(int n, LAMBDA && fun)
    {
        if (n == 0) return;

        const int block_size = 256;
        const int n_blocks = (n + block_size - 1) / block_size;

        forall_kernel<<< n_blocks, block_size >>>(n, fun);
    }

    template <typename LAMBDA>
    __global__ static void forall1d_kernel(int n, LAMBDA fun)
    {
        const int k = blockIdx.x;
        if (k >= n) return;

        fun(k);
    }

    template <typename LAMBDA>
    void forall_1d(int bx, int n, LAMBDA && fun)
    {
        if (n == 0) return;

        forall2d_kernel<<<n, bx>>>(n, fun);
    }

    template <typename LAMBDA>
    __global__ static void forall2d_kernel(int n, LAMBDA fun)
    {
        const int k = blockIdx.x;
        if (k >= n) return;

        fun(k);
    }

    template <typename LAMBDA>
    void forall_2d(int bx, int by, int n, LAMBDA && fun)
    {
        if (n == 0) return;
        
        const dim3 block_size(bx, by);
        forall2d_kernel<<<n, block_size>>>(n, fun);
    }
} // namespace cuddh

#endif
