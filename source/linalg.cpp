#include "linalg.hpp"

template <typename real_t>
struct dist_op
{
    __host__ __device__ constexpr real_t operator()(thrust::tuple<real_t, real_t> t) const
    {
        real_t x = thrust::get<0>(t);
        real_t y = thrust::get<1>(t);
        return (x - y) * (x - y);
    }
};

template <typename real_t>
struct axpby_op
{
    real_t a, b;

    __host__ __device__ real_t operator()(real_t x, real_t y) const { return a * x + b * y; }
};

namespace cuddh::dla
{
    void axpby(int n, double a, const double *x, double b, double *y)
    {
        auto px = thrust::device_pointer_cast(x);
        auto py = thrust::device_pointer_cast(y);

        thrust::transform(px, px + n, py, py, axpby_op<double>{a, b});
    }

    void axpby(int n, float a, const float *__restrict__ x, float b, float *__restrict__ y)
    {
        auto px = thrust::device_pointer_cast(x);
        auto py = thrust::device_pointer_cast(y);

        thrust::transform(px, px + n, py, py, axpby_op<float>{a, b});
    }

    double dot(int n, const double *x, const double *y)
    {
        auto px = thrust::device_pointer_cast(x);
        auto py = thrust::device_pointer_cast(y);

        return thrust::inner_product(thrust::device, px, px + n, py, 0.0);
    }

    float dot(int n, const float *x, const float *y)
    {
        auto px = thrust::device_pointer_cast(x);
        auto py = thrust::device_pointer_cast(y);

        return thrust::inner_product(thrust::device, px, px + n, py, 0.0f);
    }

    double dist(int n, const double *x, const double *y)
    {
        auto px = thrust::device_pointer_cast(x);
        auto py = thrust::device_pointer_cast(y);

        auto begin = thrust::make_zip_iterator(thrust::make_tuple(px, py));
        auto end = thrust::make_zip_iterator(thrust::make_tuple(px + n, py + n));

        return std::sqrt(thrust::transform_reduce(begin, end, dist_op<double>{}, 0.0, thrust::plus<double>()));
    }

    float dist(int n, const float *x, const float *y)
    {
        auto px = thrust::device_pointer_cast(x);
        auto py = thrust::device_pointer_cast(y);

        auto begin = thrust::make_zip_iterator(thrust::make_tuple(px, py));
        auto end = thrust::make_zip_iterator(thrust::make_tuple(px + n, py + n));

        return std::sqrt(thrust::transform_reduce(begin, end, dist_op<float>{}, 0.0f, thrust::plus<float>()));
    }

    void copy(int n, const double *x, double *y)
    {
        auto px = thrust::device_pointer_cast(x);
        auto py = thrust::device_pointer_cast(y);
        thrust::copy(px, px + n, py);
    }

    void copy(int n, const float *x, float *y)
    {
        auto px = thrust::device_pointer_cast(x);
        auto py = thrust::device_pointer_cast(y);
        thrust::copy(px, px + n, py);
    }

    void copy(int n, const int *x, int *y)
    {
        auto px = thrust::device_pointer_cast(x);
        auto py = thrust::device_pointer_cast(y);
        thrust::copy(px, px + n, py);
    }

    void scal(int n, double a, double *x)
    {
        forall(n, [=] __device__(int i) -> void { x[i] *= a; });
    }

    void scal(int n, float a, float *x)
    {
        forall(n, [=] __device__(int i) -> void { x[i] *= a; });
    }

    void fill(int n, double a, double *x)
    {
        auto px = thrust::device_pointer_cast(x);
        thrust::fill(px, px + n, a);
    }

    void fill(int n, float a, float *x)
    {
        auto px = thrust::device_pointer_cast(x);
        thrust::fill(px, px + n, a);
    }

    void fill(int n, int a, int *x)
    {
        auto px = thrust::device_pointer_cast(x);
        thrust::fill(px, px + n, a);
    }
} // namespace cuddh::dla
