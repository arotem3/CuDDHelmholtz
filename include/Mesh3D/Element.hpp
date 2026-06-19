#pragma once

#include "SmallMatrix.hpp"
#include "cuddh_config.hpp"

namespace cuddh
{
    class HexElement
    {
    public:
        using vec = double3;
        using mat = double3x3;

        __host__ __device__ HexElement() = default;
        ~HexElement() = default;

        constexpr explicit HexElement(const vec *X) : x{}
        {
            for (int i = 0; i < 8; ++i)
                x[i] = X[i];
        }

        /// @brief maps reference coordinates to physical coordinates.
        constexpr vec physical_coordinates(const vec &s) const
        {
            vec coo{0.0, 0.0, 0.0};
            for (int i = 0; i < 8; ++i)
            {
                const double p = _shape(i, s);
                coo.x += p * x[i].x;
                coo.y += p * x[i].y;
                coo.z += p * x[i].z;
            }
            return coo;
        }

        /**
         * @brief computes the Jacobian of the element mapping from reference
         * coordinates to physical coordinates.
         */
        constexpr mat jacobian(const vec &s) const
        {
            mat J{};
            for (int i = 0; i < 8; ++i)
            {
                const vec grad = _shape_grad(i, s);

                J(0, 0) += grad.x * x[i].x;
                J(0, 1) += grad.x * x[i].y;
                J(0, 2) += grad.x * x[i].z;

                J(1, 0) += grad.y * x[i].x;
                J(1, 1) += grad.y * x[i].y;
                J(1, 2) += grad.y * x[i].z;

                J(2, 0) += grad.z * x[i].x;
                J(2, 1) += grad.z * x[i].y;
                J(2, 2) += grad.z * x[i].z;
            }
            return J;
        }

        /**
         * @brief computes the determinant of the jacobian of element mapping
         * from reference coordinates to physical coordinates.
         */
        constexpr double measure(const vec &s) const { return det(jacobian(s)); }

    private:
        vec x[8];

        static constexpr double _shape(int i, const vec &s)
        {
            const double xi = (i == 0 || i == 3 || i == 4 || i == 7) ? -1.0 : 1.0;
            const double yi = (i == 0 || i == 1 || i == 4 || i == 5) ? -1.0 : 1.0;
            const double zi = (i < 4) ? -1.0 : 1.0;
            return 0.125 * (1.0 + xi * s.x) * (1.0 + yi * s.y) * (1.0 + zi * s.z);
        }

        static constexpr vec _shape_grad(int i, const vec &s)
        {
            const double xi = (i == 0 || i == 3 || i == 4 || i == 7) ? -1.0 : 1.0;
            const double yi = (i == 0 || i == 1 || i == 4 || i == 5) ? -1.0 : 1.0;
            const double zi = (i < 4) ? -1.0 : 1.0;
            return vec{0.125 * xi * (1.0 + yi * s.y) * (1.0 + zi * s.z),
                       0.125 * yi * (1.0 + xi * s.x) * (1.0 + zi * s.z),
                       0.125 * zi * (1.0 + xi * s.x) * (1.0 + yi * s.y)};
        }
    };

} // namespace cuddh
