#ifndef CUDDH_MESH_2D_HPP
#define CUDDH_MESH_2D_HPP

#include "Tensor.hpp"

namespace cuddh
{
    /// @brief Two dimensional mesh. Right now only supporting structured
    /// rectangular mesh with uniform elements
    class Mesh2D
    {
    public:
        // construct uniform mesh on [ax, bx] x [ay, by] of nx x ny elements.
        Mesh2D(int nx, double ax, double bx, int ny, double ay, double by);

        /// @brief returns the uniform jacobian determinant of the elements.
        /// jacobian = dx * dy
        double jacobian() const;

    private:

    };
} // namespace cuddh


#endif