#pragma once

#include <cuda_runtime.h>

#include <cstddef>

#include "cuddh_config.hpp"
#include "cuddh_error.hpp"

namespace cuddh
{
    template <typename T, size_t Rows, size_t Cols>
    class SmallMatrix;

    template <typename T, size_t Rows>
    class SmallSymmetricMatrix;

    template <typename T, size_t Rows, size_t Cols>
    class SmallMatrix
    {
    public:
        constexpr SmallMatrix() : data{} {}
        constexpr SmallMatrix(const SmallMatrix &other) = default;
        constexpr SmallMatrix(SmallMatrix &&other) = default;
        constexpr SmallMatrix &operator=(const SmallMatrix &other) = default;
        constexpr SmallMatrix &operator=(SmallMatrix &&other) = default;

        constexpr SmallMatrix(const SmallSymmetricMatrix<T, Rows> &other)
            requires(Rows == Cols)
        {
            for (size_t i = 0; i < Rows; i++)
                for (size_t j = 0; j < Rows; j++)
                    data[i][j] = other(i, j);
        }

        constexpr SmallMatrix &operator=(const SmallSymmetricMatrix<T, Rows> &other)
            requires(Rows == Cols)
        {
            for (size_t i = 0; i < Rows; i++)
                for (size_t j = 0; j < Rows; j++)
                    data[i][j] = other(i, j);
            return *this;
        }

        static constexpr size_t n_rows() { return Rows; }
        static constexpr size_t n_cols() { return Cols; }

        constexpr inline T &operator()(size_t i, size_t j)
        {
            cuddh_assert(i < Rows && j < Cols,
                         printf("SmallMatrix::operator() error: index out of range. i=%zu, j=%zu, Rows=%zu, Cols=%zu\n",
                                i, j, Rows, Cols));
            return data[i][j];
        }

        constexpr const T &operator()(size_t i, size_t j) const
        {
            cuddh_assert(i < Rows && j < Cols,
                         printf("SmallMatrix::operator() error: index out of range. i=%zu, j=%zu, Rows=%zu, Cols=%zu\n",
                                i, j, Rows, Cols));
            return data[i][j];
        }

        constexpr void zeros()
        {
            for (size_t i = 0; i < Rows; i++)
                for (size_t j = 0; j < Cols; j++)
                    data[i][j] = T();
        }

    private:
        T data[Rows][Cols];
    };

    template <typename T, size_t Rows>
    class SmallSymmetricMatrix
    {
    public:
        constexpr SmallSymmetricMatrix() : data{} {}
        constexpr SmallSymmetricMatrix(const SmallSymmetricMatrix &other) = default;
        constexpr SmallSymmetricMatrix(SmallSymmetricMatrix &&other) = default;
        constexpr SmallSymmetricMatrix &operator=(const SmallSymmetricMatrix &other) = default;
        constexpr SmallSymmetricMatrix &operator=(SmallSymmetricMatrix &&other) = default;

        constexpr SmallSymmetricMatrix(const SmallMatrix<T, Rows, Rows> &other)
        {
            for (size_t i = 0; i < Rows; i++)
                for (size_t j = 0; j <= i; j++)
                    data[index(i, j)] = other(i, j);
        }

        constexpr SmallSymmetricMatrix &operator=(const SmallMatrix<T, Rows, Rows> &other)
        {
            for (size_t i = 0; i < Rows; i++)
                for (size_t j = 0; j <= i; j++)
                    data[index(i, j)] = other(i, j);
            return *this;
        }

        static constexpr size_t n_rows() { return Rows; }
        static constexpr size_t n_cols() { return Rows; }

        constexpr T &operator()(size_t i, size_t j) { return data[index(i, j)]; }

        constexpr const T &operator()(size_t i, size_t j) const { return data[index(i, j)]; }

        constexpr void zeros()
        {
            for (size_t i = 0; i < Rows * (Rows + 1) / 2; i++)
                data[i] = T();
        }

    private:
        T data[Rows * (Rows + 1) / 2];

        static constexpr size_t index(size_t i, size_t j)
        {
            cuddh_assert(i < Rows && j < Rows,
                         printf("SmallSymmetricMatrix::index() error: index out of range. i=%zu, j=%zu, Rows=%zu\n", i,
                                j, Rows));

            if (i < j)
                return j * (j + 1) / 2 + i;
            else
                return i * (i + 1) / 2 + j;
        }
    };

    using double2x2 = SmallMatrix<double, 2, 2>;
    using double3x3 = SmallMatrix<double, 3, 3>;

    using float2x2 = SmallMatrix<float, 2, 2>;
    using float3x3 = SmallMatrix<float, 3, 3>;

    using dsym2x2 = SmallSymmetricMatrix<double, 2>;
    using dsym3x3 = SmallSymmetricMatrix<double, 3>;

    using fsym2x2 = SmallSymmetricMatrix<float, 2>;
    using fsym3x3 = SmallSymmetricMatrix<float, 3>;

    using double3x2 = SmallMatrix<double, 3, 2>;
    using float3x2 = SmallMatrix<float, 3, 2>;

    template <typename T>
    using scalar2 = std::conditional_t<std::is_same_v<T, float>, float2, double2>;

    template <typename T>
    using scalar3 = std::conditional_t<std::is_same_v<T, float>, float3, double3>;

    constexpr double2 operator*(const double2x2 &A, const double2 &x)
    {
        double2 y;
        y.x = A(0, 0) * x.x + A(0, 1) * x.y;
        y.y = A(1, 0) * x.x + A(1, 1) * x.y;
        return y;
    }

    constexpr double3 operator*(const double3x3 &A, const double3 &x)
    {
        double3 y;
        y.x = A(0, 0) * x.x + A(0, 1) * x.y + A(0, 2) * x.z;
        y.y = A(1, 0) * x.x + A(1, 1) * x.y + A(1, 2) * x.z;
        y.z = A(2, 0) * x.x + A(2, 1) * x.y + A(2, 2) * x.z;
        return y;
    }

    constexpr double2 operator*(const dsym2x2 &A, const double2 &x)
    {
        double2 y;
        y.x = A(0, 0) * x.x + A(0, 1) * x.y;
        y.y = A(1, 0) * x.x + A(1, 1) * x.y;
        return y;
    }

    constexpr double3 operator*(const dsym3x3 &A, const double3 &x)
    {
        double3 y;
        y.x = A(0, 0) * x.x + A(0, 1) * x.y + A(0, 2) * x.z;
        y.y = A(1, 0) * x.x + A(1, 1) * x.y + A(1, 2) * x.z;
        y.z = A(2, 0) * x.x + A(2, 1) * x.y + A(2, 2) * x.z;
        return y;
    }

    constexpr float2 operator*(const float2x2 &A, const float2 &x)
    {
        float2 y;
        y.x = A(0, 0) * x.x + A(0, 1) * x.y;
        y.y = A(1, 0) * x.x + A(1, 1) * x.y;
        return y;
    }

    constexpr float3 operator*(const float3x3 &A, const float3 &x)
    {
        float3 y;
        y.x = A(0, 0) * x.x + A(0, 1) * x.y + A(0, 2) * x.z;
        y.y = A(1, 0) * x.x + A(1, 1) * x.y + A(1, 2) * x.z;
        y.z = A(2, 0) * x.x + A(2, 1) * x.y + A(2, 2) * x.z;
        return y;
    }

    constexpr float2 operator*(const fsym2x2 &A, const float2 &x)
    {
        float2 y;
        y.x = A(0, 0) * x.x + A(0, 1) * x.y;
        y.y = A(1, 0) * x.x + A(1, 1) * x.y;
        return y;
    }

    constexpr float3 operator*(const fsym3x3 &A, const float3 &x)
    {
        float3 y;
        y.x = A(0, 0) * x.x + A(0, 1) * x.y + A(0, 2) * x.z;
        y.y = A(1, 0) * x.x + A(1, 1) * x.y + A(1, 2) * x.z;
        y.z = A(2, 0) * x.x + A(2, 1) * x.y + A(2, 2) * x.z;
        return y;
    }

    constexpr double det(const double2x2 &A)
    {
        return A(0, 0) * A(1, 1) - A(0, 1) * A(1, 0);
    }

    constexpr double det(const double3x3 &A)
    {
        return A(0, 0) * (A(1, 1) * A(2, 2) - A(1, 2) * A(2, 1)) - A(0, 1) * (A(1, 0) * A(2, 2) - A(1, 2) * A(2, 0)) +
               A(0, 2) * (A(1, 0) * A(2, 1) - A(1, 1) * A(2, 0));
    }

    constexpr double det(const dsym2x2 &A)
    {
        return A(0, 0) * A(1, 1) - A(0, 1) * A(0, 1);
    }

    constexpr double det(const dsym3x3 &A)
    {
        return A(0, 0) * (A(1, 1) * A(2, 2) - A(1, 2) * A(1, 2)) - A(0, 1) * (A(1, 0) * A(2, 2) - A(1, 2) * A(0, 2)) +
               A(0, 2) * (A(1, 0) * A(1, 1) - A(0, 1) * A(0, 2));
    }

    constexpr float det(const float2x2 &A)
    {
        return A(0, 0) * A(1, 1) - A(0, 1) * A(1, 0);
    }

    constexpr float det(const float3x3 &A)
    {
        return A(0, 0) * (A(1, 1) * A(2, 2) - A(1, 2) * A(2, 1)) - A(0, 1) * (A(1, 0) * A(2, 2) - A(1, 2) * A(2, 0)) +
               A(0, 2) * (A(1, 0) * A(2, 1) - A(1, 1) * A(2, 0));
    }

    constexpr float det(const fsym2x2 &A)
    {
        return A(0, 0) * A(1, 1) - A(0, 1) * A(0, 1);
    }

    constexpr float det(const fsym3x3 &A)
    {
        return A(0, 0) * (A(1, 1) * A(2, 2) - A(1, 2) * A(1, 2)) - A(0, 1) * (A(1, 0) * A(2, 2) - A(1, 2) * A(0, 2)) +
               A(0, 2) * (A(1, 0) * A(1, 1) - A(0, 1) * A(0, 2));
    }

    constexpr double2x2 adjugate(const double2x2 &A)
    {
        double2x2 B;
        B(0, 0) = A(1, 1);
        B(0, 1) = -A(0, 1);
        B(1, 0) = -A(1, 0);
        B(1, 1) = A(0, 0);
        return B;
    }

    constexpr float2x2 adjugate(const float2x2 &A)
    {
        float2x2 B;
        B(0, 0) = A(1, 1);
        B(0, 1) = -A(0, 1);
        B(1, 0) = -A(1, 0);
        B(1, 1) = A(0, 0);
        return B;
    }

    constexpr dsym2x2 adjugate(const dsym2x2 &A)
    {
        dsym2x2 B;
        B(0, 0) = A(1, 1);
        B(0, 1) = -A(0, 1);
        B(1, 1) = A(0, 0);
        return B;
    }

    constexpr fsym2x2 adjugate(const fsym2x2 &A)
    {
        fsym2x2 B;
        B(0, 0) = A(1, 1);
        B(0, 1) = -A(0, 1);
        B(1, 1) = A(0, 0);
        return B;
    }

    constexpr double3x3 adjugate(const double3x3 &A)
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

    constexpr float3x3 adjugate(const float3x3 &A)
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

    constexpr dsym3x3 adjugate(const dsym3x3 &A)
    {
        dsym3x3 B;
        B(0, 0) = A(1, 1) * A(2, 2) - A(1, 2) * A(1, 2);
        B(0, 1) = A(0, 2) * A(2, 1) - A(0, 1) * A(2, 2);
        B(0, 2) = A(0, 1) * A(1, 2) - A(0, 2) * A(1, 1);
        B(1, 1) = A(0, 0) * A(2, 2) - A(0, 2) * A(2, 0);
        B(1, 2) = A(0, 2) * A(1, 0) - A(0, 0) * A(1, 2);
        B(2, 2) = A(0, 0) * A(1, 1) - A(0, 1) * A(1, 0);
        return B;
    }

    constexpr fsym3x3 adjugate(const fsym3x3 &A)
    {
        fsym3x3 B;
        B(0, 0) = A(1, 1) * A(2, 2) - A(1, 2) * A(1, 2);
        B(0, 1) = A(0, 2) * A(2, 1) - A(0, 1) * A(2, 2);
        B(0, 2) = A(0, 1) * A(1, 2) - A(0, 2) * A(1, 1);
        B(1, 1) = A(0, 0) * A(2, 2) - A(0, 2) * A(2, 0);
        B(1, 2) = A(0, 2) * A(1, 0) - A(0, 0) * A(1, 2);
        B(2, 2) = A(0, 0) * A(1, 1) - A(0, 1) * A(1, 0);
        return B;
    }

    constexpr double2x2 transpose(const double2x2 &A)
    {
        double2x2 B;
        B(0, 0) = A(0, 0);
        B(0, 1) = A(1, 0);
        B(1, 0) = A(0, 1);
        B(1, 1) = A(1, 1);
        return B;
    }

    constexpr double3x3 transpose(const double3x3 &A)
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

    constexpr float2x2 transpose(const float2x2 &A)
    {
        float2x2 B;
        B(0, 0) = A(0, 0);
        B(0, 1) = A(1, 0);
        B(1, 0) = A(0, 1);
        B(1, 1) = A(1, 1);
        return B;
    }

    constexpr float3x3 transpose(const float3x3 &A)
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
