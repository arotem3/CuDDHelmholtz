#pragma once

#include <cuda_runtime.h>

#include "cuddh_config.hpp"
#include "cuddh_error.hpp"

namespace cuddh
{
    template <typename scalar, size_t... Shape>
    class FixedTensorWrapper
    {
    public:
        __host__ __device__ static inline constexpr size_t size() { return (Shape * ...); }

        __host__ __device__ static inline constexpr size_t shape(size_t dim)
        {
            return (dim < sizeof...(Shape)) ? ((size_t[]){Shape...})[dim] : 0;
        }

        __host__ __device__ inline constexpr scalar *data() { return ptr; }

        __host__ __device__ inline constexpr const scalar *data() const { return ptr; }

        __host__ __device__ inline constexpr FixedTensorWrapper(scalar *data = nullptr) : ptr(data) {}

        inline constexpr FixedTensorWrapper(const FixedTensorWrapper &other) = default;
        inline constexpr FixedTensorWrapper &operator=(const FixedTensorWrapper &other) = default;
        inline constexpr FixedTensorWrapper(FixedTensorWrapper &&other) = default;
        inline constexpr FixedTensorWrapper &operator=(FixedTensorWrapper &&other) = default;

        template <typename... Indices>
        __host__ __device__ inline constexpr scalar &at(Indices... ids)
        {
            static_assert(sizeof...(ids) == sizeof...(Shape), "Wrong number of indices specified.");
            return ptr[compute_index<Shape...>(ids...)];
        }

        template <typename... Indices>
        __host__ __device__ inline constexpr const scalar &at(Indices... ids) const
        {
            static_assert(sizeof...(ids) == sizeof...(Shape), "Wrong number of indices specified.");
            return ptr[compute_index<Shape...>(ids...)];
        }

        template <typename... Indices>
        __host__ __device__ inline constexpr scalar &operator()(Indices... ids)
        {
            return at(std::forward<Indices>(ids)...);
        }

        template <typename... Indices>
        __host__ __device__ inline constexpr const scalar &operator()(Indices... ids) const
        {
            return at(std::forward<Indices>(ids)...);
        }

    private:
        scalar *ptr;

        template <size_t N, size_t... Ns, typename... Inds>
        __host__ __device__ static inline constexpr int compute_index(int index, Inds... ids)
        {
            if constexpr (sizeof...(Ns) == 0)
                return index;
            else
                return index + N * compute_index<Ns...>(ids...);
        }
    };

    template <size_t... Shape, typename scalar>
    __host__ __device__ inline constexpr auto make_fixed_view(scalar *data)
    {
        return FixedTensorWrapper<scalar, Shape...>(data);
    }

    template <typename scalar, size_t... Shape>
    class FixedTensor
    {
    public:
        inline constexpr FixedTensor() = default;
        inline constexpr FixedTensor(const FixedTensor &other) = default;
        inline constexpr FixedTensor &operator=(const FixedTensor &other) = default;
        inline constexpr FixedTensor(FixedTensor &&other) = default;
        inline constexpr FixedTensor &operator=(FixedTensor &&other) = default;

        __host__ __device__ static inline constexpr size_t size() { return (Shape * ...); }

        __host__ __device__ static inline constexpr size_t shape(size_t dim)
        {
            return (dim < sizeof...(Shape)) ? ((size_t[]){Shape...})[dim] : 0;
        }

        __host__ __device__ inline constexpr scalar *data() { return ptr; }

        __host__ __device__ inline constexpr const scalar *data() const { return ptr; }

        template <typename... Indices>
        __host__ __device__ inline constexpr scalar &at(Indices... ids)
        {
            static_assert(sizeof...(ids) == sizeof...(Shape), "Wrong number of indices specified.");
            return ptr[compute_index<Shape...>(ids...)];
        }

        template <typename... Indices>
        __host__ __device__ inline constexpr const scalar &at(Indices... ids) const
        {
            static_assert(sizeof...(ids) == sizeof...(Shape), "Wrong number of indices specified.");
            return ptr[compute_index<Shape...>(ids...)];
        }

        template <typename... Indices>
        __host__ __device__ inline constexpr scalar &operator()(Indices... ids)
        {
            return at(std::forward<Indices>(ids)...);
        }

        template <typename... Indices>
        __host__ __device__ inline constexpr const scalar &operator()(Indices... ids) const
        {
            return at(std::forward<Indices>(ids)...);
        }

        __host__ __device__ inline constexpr scalar &operator[](int idx)
        {
            cuddh_assert(0 <= idx && idx < size(),
                         printf("FixedTensor index out of bounds: %d not in [0, %zu)\n", idx, size()););
            return ptr[idx];
        }

        __host__ __device__ inline constexpr const scalar &operator[](int idx) const
        {
            cuddh_assert(0 <= idx && idx < size(),
                         printf("FixedTensor index out of bounds: %d not in [0, %zu)\n", idx, size()););
            return ptr[idx];
        }

    private:
        scalar ptr[(1 * ... * Shape)];

        template <size_t N, size_t... Ns, typename... Inds>
        __host__ __device__ static inline constexpr int compute_index(int index, Inds... ids)
        {
            cuddh_assert(0 <= index && index < N,
                         printf("FixedTensor index out of bounds: %d not in [0, %zu)\n", index, N););
            if constexpr (sizeof...(Ns) == 0)
                return index;
            else
                return index + N * compute_index<Ns...>(ids...);
        }
    };
} // namespace cuddh
