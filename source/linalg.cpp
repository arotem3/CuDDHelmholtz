#include "linalg.hpp"

// The reduction kernel performs a binary tree summation reduction. SZ is the
// block size and NR is the number of reads performed by each thread.
// result = sum_k op(k, x, y). e.g. for dot product op(k, x, y) = x[k] * y[k]
template <int SZ, int NR, typename scalar, typename LAMBDA>
__global__ static void sum_reduction_kernel(int n, const scalar * x, const scalar * y, scalar * __restrict__ result, LAMBDA op)
{
#ifndef CUDDH_DEBUG
    const int block_dim = blockDim.x;
    assert(block_dim == SZ);
#endif

	const int thread_id = threadIdx.x;
    const int block_id = blockIdx.x;

    __shared__ scalar s[SZ];

    scalar sum = 0.0;

    #pragma unroll
    for (int j = 0; j < NR; ++j)
    {
        const int k = thread_id + SZ * (j + NR * block_id);
        if (k < n)
            sum += op(k, x, y);
    }

    s[thread_id] = sum;

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
        sum = s[0];
        atomicAdd(result, sum);
    }
}

namespace cuddh
{
    void axpby(int n, double a, const double * __restrict__ x, double b, double * __restrict__ y)
    {
        forall(n, [=] __device__ (int i) -> void
        {
            y[i] = a * x[i] + b * y[i];
        });
    }

    void axpby(int n, float a, const float * __restrict__ x, float b, float * __restrict__ y)
    {
        forall(n, [=] __device__ (int i) -> void
        {
            y[i] = a * x[i] + b * y[i];
        });
    }

    double dot(int n, const double * x, const double * y)
    {
        host_device_dvec result(1);

        constexpr int block_size = 32;
        constexpr int num_reads = 8;
        constexpr int data_per_block = block_size * num_reads;

        const int n_blocks = (n + data_per_block - 1) / data_per_block;

        double * d_result = result.device_read_write();
        zeros(1, d_result);
        
        sum_reduction_kernel<block_size, num_reads, double> <<< n_blocks, block_size >>>(n, x, y, d_result, [] __device__ (int k, const double * X, const double * Y) {return X[k] * Y[k];});

        return *result.host_read();
    }

    float dot(int n, const float * x, const float * y)
    {
        HostDeviceArray<float> result(1);

        constexpr int block_size = 32;
        constexpr int num_reads = 8;
        constexpr int data_per_block = block_size * num_reads;

        const int n_blocks = (n + data_per_block - 1) / data_per_block;

        float * d_result = result.device_read_write();
        zeros(1, d_result);
        
        sum_reduction_kernel<block_size, num_reads, float> <<< n_blocks, block_size >>>(n, x, y, d_result, [] __device__ (int k, const float * X, const float * Y) {return X[k] * Y[k];});

        return *result.host_read();
    }

    double dist(int n, const double * x, const double * y)
    {
        host_device_dvec result(1);

        constexpr int block_size = 32;
        constexpr int num_reads = 8;
        constexpr int data_per_block = block_size * num_reads;

        const int n_blocks = (n + data_per_block - 1) / data_per_block;

        double * d_result = result.device_read_write();
        zeros(1, d_result);

        sum_reduction_kernel<block_size, num_reads> <<< n_blocks, block_size >>>(n, x, y, d_result, [] __device__ (int k, const double * X, const double * Y) {double e = X[k]-Y[k]; return e*e;});

        return std::sqrt(*result.host_read());
    }

    float dist(int n, const float * x, const float * y)
    {
        HostDeviceArray<float> result(1);

        constexpr int block_size = 32;
        constexpr int num_reads = 8;
        constexpr int data_per_block = block_size * num_reads;

        const int n_blocks = (n + data_per_block - 1) / data_per_block;

        float * d_result = result.device_read_write();
        zeros(1, d_result);

        sum_reduction_kernel<block_size, num_reads, float> <<< n_blocks, block_size >>>(n, x, y, d_result, [] __device__ (int k, const float * X, const float * Y) {float e = X[k]-Y[k]; return e*e;});

        return std::sqrt(*result.host_read());
    }

    void copy(int n, const double * __restrict__ x, double * __restrict__ y)
    {
        forall(n, [=] __device__ (int i) -> void
        {
            y[i] = x[i];
        });
    }

    void copy(int n, const float * __restrict__ x, float * __restrict__ y)
    {
        forall(n, [=] __device__ (int i) -> void
        {
            y[i] = x[i];
        });
    }

    void copy(int n, const int * __restrict__ x, int * __restrict__ y)
    {
        forall(n, [=] __device__ (int i) -> void
        {
            y[i] = x[i];
        });
    }

    void scal(int n, double a, double * x)
    {
        forall(n, [=] __device__ (int i) -> void
        {
            x[i] *= a;
        });
    }

    void scal(int n, float a, float * x)
    {
        forall(n, [=] __device__ (int i) -> void
        {
            x[i] *= a;
        });
    }

    void fill(int n, double a, double * x)
    {
        forall(n, [=] __device__ (int i) -> void
        {
            x[i] = a;
        });
    }

    void fill(int n, float a, float * x)
    {
        forall(n, [=] __device__ (int i) -> void
        {
            x[i] = a;
        });
    }

    void fill(int n, int a, int * x)
    {
        forall(n, [=] __device__ (int i) -> void
        {
            x[i] = a;
        });
    }
} // namespace cuddh
