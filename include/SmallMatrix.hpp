#ifndef CUDDH_SMALL_MATRIX_HPP
#define CUDDH_SMALL_MATRIX_HPP

#include "cuddh_config.hpp"
#include "cuddh_error.hpp"
#include <cuda_runtime.h>

namespace cuddh
{
  template <typename T, size_t Rows, size_t Cols>
  class SmallMatrix
  {
  public:
    constexpr SmallMatrix() = default;
    constexpr SmallMatrix(const SmallMatrix &other) = default;
    constexpr SmallMatrix(SmallMatrix &&other) = default;
    constexpr SmallMatrix &operator=(const SmallMatrix &other) = default;
    constexpr SmallMatrix &operator=(SmallMatrix &&other) = default;

    __host__ __device__ inline T &operator()(size_t i, size_t j) 
    { 
#ifdef CUDDH_DEBUG
      if (i < 0 || i >= Rows || j < 0 || j >= Cols)
        cuddh_error("SmallMatrix::operator() error: index out of range.");
#endif
      return data[i][j];
    }

    __host__ __device__ inline const T &operator()(size_t i, size_t j) const
    {
#ifdef CUDDH_DEBUG
      if (i < 0 || i >= Rows || j < 0 || j >= Cols)
        cuddh_error("SmallMatrix::operator() error: index out of range.");
#endif
      return data[i][j];
    }

    __host__ __device__ static size_t n_rows() { return Rows; }
    __host__ __device__ static size_t n_cols() { return Cols; }
  
  private:
    T data[Rows][Cols];
  };

  using double2x2 = SmallMatrix<double, 2, 2>;
  using double3x3 = SmallMatrix<double, 3, 3>;
  using float2x2 = SmallMatrix<float, 2, 2>;
  using float3x3 = SmallMatrix<float, 3, 3>;

  using double3x2 = SmallMatrix<double, 3, 2>;
  using float3x2 = SmallMatrix<float, 3, 2>;

  template <typename T, size_t Rows, size_t Cols>
  __host__ __device__
  inline void zeros(SmallMatrix<T, Rows, Cols> &A)
  {
    for (size_t i = 0; i < Rows; i++)
      for (size_t j = 0; j < Cols; j++)
        A(i, j) = T();
  }

  __host__ __device__
  inline double2 operator*(const double2x2 &A, const double2 &x)
  {
    double2 y;
    y.x = A(0, 0) * x.x + A(0, 1) * x.y;
    y.y = A(1, 0) * x.x + A(1, 1) * x.y;
    return y;
  }

  __host__ __device__
  inline double3 operator*(const double3x3 &A, const double3 &x)
  {
    double3 y;
    y.x = A(0, 0) * x.x + A(0, 1) * x.y + A(0, 2) * x.z;
    y.y = A(1, 0) * x.x + A(1, 1) * x.y + A(1, 2) * x.z;
    y.z = A(2, 0) * x.x + A(2, 1) * x.y + A(2, 2) * x.z;
    return y;
  }

  __host__ __device__
  inline float2 operator*(const float2x2 &A, const float2 &x)
  {
    float2 y;
    y.x = A(0, 0) * x.x + A(0, 1) * x.y;
    y.y = A(1, 0) * x.x + A(1, 1) * x.y;
    return y;
  }

  __host__ __device__
  inline float3 operator*(const float3x3 &A, const float3 &x)
  {
    float3 y;
    y.x = A(0, 0) * x.x + A(0, 1) * x.y + A(0, 2) * x.z;
    y.y = A(1, 0) * x.x + A(1, 1) * x.y + A(1, 2) * x.z;
    y.z = A(2, 0) * x.x + A(2, 1) * x.y + A(2, 2) * x.z;
    return y;
  }
  
  __host__ __device__
  inline double det(const double2x2 &A)
  {
    return A(0, 0) * A(1, 1) - A(0, 1) * A(1, 0);
  }

  __host__ __device__
  inline double det(const double3x3 &A)
  {
    return A(0, 0) * (A(1, 1) * A(2, 2) - A(1, 2) * A(2, 1)) -
           A(0, 1) * (A(1, 0) * A(2, 2) - A(1, 2) * A(2, 0)) +
           A(0, 2) * (A(1, 0) * A(2, 1) - A(1, 1) * A(2, 0));
  }

  __host__ __device__
  inline float det(const float2x2 &A)
  {
    return A(0, 0) * A(1, 1) - A(0, 1) * A(1, 0);
  }

  __host__ __device__
  inline float det(const float3x3 &A)
  {
    return A(0, 0) * (A(1, 1) * A(2, 2) - A(1, 2) * A(2, 1)) -
           A(0, 1) * (A(1, 0) * A(2, 2) - A(1, 2) * A(2, 0)) +
           A(0, 2) * (A(1, 0) * A(2, 1) - A(1, 1) * A(2, 0));
  }

  __host__ __device__
  inline double2x2 adjugate(const double2x2 &A)
  {
    double2x2 B;
    B(0, 0) = A(1, 1);
    B(0, 1) = -A(0, 1);
    B(1, 0) = -A(1, 0);
    B(1, 1) = A(0, 0);
    return B;
  }

  __host__ __device__
  inline float2x2 adjugate(const float2x2 &A)
  {
    float2x2 B;
    B(0, 0) = A(1, 1);
    B(0, 1) = -A(0, 1);
    B(1, 0) = -A(1, 0);
    B(1, 1) = A(0, 0);
    return B;
  }

  __host__ __device__
  inline double3x3 adjugate(const double3x3 &A)
  {
    double3x3 B;
    B(0, 0) = A(1, 1) * A(2, 2) - A(1, 2) * A(2, 1);
    B(0, 1) = A(0, 2) * A(2, 1) - A(0, 1) * A(2, 2);
    B(0, 2) = A(0, 1) * A(1, 2) - A(0, 2) * A(1, 1);
    B(1, 0) = A(1, 2) * A(2, 0) - A(1, 0) * A(2, 2);
    B(1, 1) = A(0, 0) * A(2, 2) - A(0, 2) * A(2, 0);
    B(1, 2) = A(0, 2) * A(1, 0) - A(0, 0) * A(1, 2);
    B(2, 0) = A(1, 0) * A(2, 1) - A(1, 1) * A(2, 0);
    B(2, 1) = A(0, 1) * A(2, 0) - A(0, 0) * A(2, 1);
    B(2, 2) = A(0, 0) * A(1, 1) - A(0, 1) * A(1, 0);
    return B;
  }

  __host__ __device__
  inline float3x3 adjugate(const float3x3 &A)
  {
    float3x3 B;
    B(0, 0) = A(1, 1) * A(2, 2) - A(1, 2) * A(2, 1);
    B(0, 1) = A(0, 2) * A(2, 1) - A(0, 1) * A(2, 2);
    B(0, 2) = A(0, 1) * A(1, 2) - A(0, 2) * A(1, 1);
    B(1, 0) = A(1, 2) * A(2, 0) - A(1, 0) * A(2, 2);
    B(1, 1) = A(0, 0) * A(2, 2) - A(0, 2) * A(2, 0);
    B(1, 2) = A(0, 2) * A(1, 0) - A(0, 0) * A(1, 2);
    B(2, 0) = A(1, 0) * A(2, 1) - A(1, 1) * A(2, 0);
    B(2, 1) = A(0, 1) * A(2, 0) - A(0, 0) * A(2, 1);
    B(2, 2) = A(0, 0) * A(1, 1) - A(0, 1) * A(1, 0);
    return B;
  }

  __host__ __device__
  inline double2x2 transpose(const double2x2 &A)
  {
    double2x2 B;
    B(0, 0) = A(0, 0);
    B(0, 1) = A(1, 0);
    B(1, 0) = A(0, 1);
    B(1, 1) = A(1, 1);
    return B;
  }

  __host__ __device__
  inline double3x3 transpose(const double3x3 &A)
  {
    double3x3 B;
    B(0, 0) = A(0, 0);
    B(0, 1) = A(1, 0);
    B(0, 2) = A(2, 0);
    B(1, 0) = A(0, 1);
    B(1, 1) = A(1, 1);
    B(1, 2) = A(2, 1);
    B(2, 0) = A(0, 2);
    B(2, 1) = A(1, 2);
    B(2, 2) = A(2, 2);
    return B;
  }

  __host__ __device__
  inline float2x2 transpose(const float2x2 &A)
  {
    float2x2 B;
    B(0, 0) = A(0, 0);
    B(0, 1) = A(1, 0);
    B(1, 0) = A(0, 1);
    B(1, 1) = A(1, 1);
    return B;
  }

  __host__ __device__
  inline float3x3 transpose(const float3x3 &A)
  {
    float3x3 B;
    B(0, 0) = A(0, 0);
    B(0, 1) = A(1, 0);
    B(0, 2) = A(2, 0);
    B(1, 0) = A(0, 1);
    B(1, 1) = A(1, 1);
    B(1, 2) = A(2, 1);
    B(2, 0) = A(0, 2);
    B(2, 1) = A(1, 2);
    B(2, 2) = A(2, 2);
    return B;
  }
} // namespace cuddh

#endif