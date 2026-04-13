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

        /// @brief maps reference coordinates to physical coordinates.
        __host__ __device__ vec physical_coordinates(const vec &xi) const;

        /**
         * @brief computes the Jacobian of the element mapping from reference
         * coordinates to physical coordinates.
         *
         * @param xi
         * @param J
         */
        __host__ __device__ mat jacobian(const vec &xi) const;

        /**
         * @brief computes the determinant of the jacobian of element mapping
         * from reference coordinates to physical coordinates.
         *
         * @param xi
         * @return double
         */
        __host__ __device__ double measure(const vec &xi) const;

        __host__ __device__ HexElement(const vec *X);

        HexElement() = default;
        ~HexElement() = default;

    private:
        vec x[8]; // nodes are ordered in canonical order
    };

} // namespace cuddh
