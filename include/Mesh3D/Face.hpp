#ifndef CUDDH_MESH3D_FACE_HPP
#define CUDDH_MESH3D_FACE_HPP

#include "cuddh_config.hpp"
#include <cuda_runtime.h>
#include <vector>

#include "SmallMatrix.hpp"

namespace cuddh
{
    class QuadFace
    {
    public:
        QuadFace() = default;
        ~QuadFace() = default;

        __host__ __device__ QuadFace(const double3 *X);

        /**
         * @brief Maps the reference coordinates to the physical coordinates.
         */
        __host__ __device__ double3 physical_coordinates(const double2 &xi) const;

        /**
         * @brief Computes the normal vector at the point `xi`.
         * The norm is the density of the surface area differential.
         * i.e.
         * n = \frac{\partial x}{\partial \xi_1} \times \frac{\partial x}{\partial \xi_2}
         */
        __host__ __device__ double3 weighted_normal(const double2 &xi) const;

        /**
         * @brief Computes the unit normal vector at the point `xi`.
         */
        __host__ __device__ double3 normal(const double2 &xi) const;

        /**
         * @brief Computes the Jacobian of the element mapping from reference
         * coordinates to physical coordinates.
         */
        __host__ __device__ double3x2 jacobian(const double2 &xi) const;

        /**
         * @brief Computes the determinant of the jacobian of element mapping
         * from reference coordinates to physical coordinates.
         */
        __host__ __device__ double measure(const double2 &xi) const;

    private:
        double3 x[4];
    };
} // namespace cuddh

#endif
