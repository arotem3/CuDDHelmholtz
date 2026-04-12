#ifndef CUDDH_ELEMENT_HPP
#define CUDDH_ELEMENT_HPP

#include "SmallMatrix.hpp"
#include "cuddh_config.hpp"

namespace cuddh
{
    /// @brief Bilinear quadrilateral element.
    ///
    /// The reference element is [-1,1]^2. Corner ordering (counter-clockwise):
    ///   0 = (-,-),  1 = (+,-),  2 = (+,+),  3 = (-,+).
    class QuadElement
    {
    private:
        double2 x[4];

    public:
        QuadElement() = default;
        ~QuadElement() = default;

        constexpr explicit QuadElement(const double2 *X) : x{}
        {
            for (int i = 0; i < 4; ++i)
                x[i] = X[i];
        }

        /// @brief Maps reference coordinates to physical coordinates.
        /// @param xi Reference point in [-1,1]^2.
        /// @return Physical coordinates.
        constexpr double2 physical_coordinates(double2 xi) const
        {
            const double b[] = {0.25 * (1.0 - xi.x) * (1.0 - xi.y), 0.25 * (1.0 + xi.x) * (1.0 - xi.y),
                                0.25 * (1.0 + xi.x) * (1.0 + xi.y), 0.25 * (1.0 - xi.x) * (1.0 + xi.y)};

            double2 p{};

            for (int i = 0; i < 4; ++i)
            {
                p.x += x[i].x * b[i];
                p.y += x[i].y * b[i];
            }

            return p;
        }

        /// @brief Computes the Jacobian of the element mapping.
        ///
        /// Returns J where J(row, col) = dx_row / dxi_col, i.e.
        ///   J(0,0) = dx/dxi0,  J(1,0) = dy/dxi0,
        ///   J(0,1) = dx/dxi1,  J(1,1) = dy/dxi1.
        /// @param xi Reference point in [-1,1]^2.
        /// @return 2x2 Jacobian matrix.
        constexpr double2x2 jacobian(double2 xi) const
        {
            double2x2 J{};

            J(0, 0) = 0.25 * ((1.0 - xi.y) * (x[1].x - x[0].x) + (1.0 + xi.y) * (x[2].x - x[3].x)); // dx/d(xi)
            J(1, 0) = 0.25 * ((1.0 - xi.y) * (x[1].y - x[0].y) + (1.0 + xi.y) * (x[2].y - x[3].y)); // dy/d(xi)
            J(0, 1) = 0.25 * ((1.0 - xi.x) * (x[3].x - x[0].x) + (1.0 + xi.x) * (x[2].x - x[1].x)); // dx/d(eta)
            J(1, 1) = 0.25 * ((1.0 - xi.x) * (x[3].y - x[0].y) + (1.0 + xi.x) * (x[2].y - x[1].y)); // dy/d(eta)

            return J;
        }

        /// @brief Returns det(J(xi)), the measure weight at the reference point xi.
        constexpr double measure(double2 xi) const { return det(jacobian(xi)); }
    };

} // namespace cuddh

#endif
