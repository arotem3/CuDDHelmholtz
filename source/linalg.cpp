#include "linalg.hpp"


// The ddot kernel performs the dot product via binary tree reduction. SZ is the
// block size and NR is the number of reads performed by each thread
template <int SZ, int NR>
__global__ static void ddot_kernel(int n, const double * x, const double * y, double * result)
{
#ifndef CUDDH_DEBUG
    const int block_dim = blockDim.x;
    assert(block_dim == SZ);
#endif

	const int thread_id = threadIdx.x;
    const int block_id = blockIdx.x;

    __shared__ double s[SZ];

    double dot = 0.0;

    #pragma unroll
    for (int j = 0; j < NR; ++j)
    {
        const int k = thread_id + SZ * (j + NR * block_id);
        dot += (k < n) ? ( x[k] * y[k] ) : 0.0;
    }

    s[thread_id] = dot;

    // tree reduction
    for (int m = SZ>>1; m > 0; m >>= 1)
    {
        __syncthreads();

        if (thread_id < m)
        {
            s[thread_id] += s[thread_id + m];
        }
    }

    if (thread_id == 0)
    {
        dot = s[0];
        atomicAdd(result, dot);
    }
}

namespace cuddh
{
    void axpby(int n, double a, const double * __restrict__ x, double b, double * __restrict__ y)
    {
        forall(n, [=] __device__ (int i) -> void {
            y[i] = a * x[i] + b * y[i];
        });
    }

    double norm(int n, const double * x)
    {
        double d = dot(n, x, x);
        return std::sqrt(d);
    }

    double dot(int n, const double * x, const double * y)
    {
        static host_device_dvec result(1);

        constexpr int block_size = 32;
        constexpr int num_reads = 8;
        constexpr int data_per_block = block_size * num_reads;

        const int n_blocks = (n + data_per_block - 1) / data_per_block;

        double * d_result = result.device_read_write();
        zeros(1, d_result);
        
        ddot_kernel<block_size, num_reads> <<< n_blocks, block_size >>>(n, x, y, d_result);

        return *result.host_read();
    }

    void copy(int n, const double * x, double * y)
    {
        forall(n, [=] __device__ (int i) -> void {
            y[i] = x[i];
        });
    }

    void scal(int n, double a, double * x)
    {
        forall(n, [=] __device__ (int i) -> void {
            x[i] *= a;
        });
    }

    void fill(int n, double a, double * x)
    {
        forall(n, [=] __device__ (int i) -> void {
            x[i] = a;
        });
    }

    void zeros(int n, double * x)
    {
        fill(n, 0.0, x);
    }

    void ones(int n, double * x)
    {
        fill(n, 1.0, x);
    }
} // namespace cuddh
