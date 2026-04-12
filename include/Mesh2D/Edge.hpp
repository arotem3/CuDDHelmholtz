#ifndef CUDDH_EDGE_HPP
#define CUDDH_EDGE_HPP

#include "cuddh_config.hpp"
#include "cuddh_error.hpp"

namespace cuddh
{
    enum class FaceType
    {
        INTERIOR, ///< @brief Edge is on interior of mesh.
                  ///<
                  ///< Edges on interior have elements on both labels, thus it
                  ///< is expected that `elements[1]` and `labels[1]` are defined.
        BOUNDARY  ///< @brief Edge is on boundary of mesh.
                  ///<
                  ///< Edges on the boundary have only one element connected,
                  ///< thus `elements[1]` and `labels[1]` need not be specified.
    };

    /// @brief Edge geometry.
    ///
    /// The edge maps the reference interval [-1,1] to the line segment
    /// [x0, x1]. The outward unit normal is determined geometrically as the
    /// right-hand perpendicular of the edge direction: rotating (x1 - x0) by
    /// 90° clockwise. Edge nodes must be pre-ordered so this convention gives
    /// the outward normal.
    class Edge
    {
    private:
        double2 x0;  ///< start point
        double2 dx;  ///< full displacement x1 - x0
        double2 n;   ///< unit outward normal
        double meas; ///< half-length = |dx| / 2

    public:
        Edge() = default;
        ~Edge() = default;

        /// @brief Construct from two endpoints.
        /// Edge nodes must be ordered such that the right-hand perpendicular
        /// of the edge direction points outward (for correct normal direction).
        /// @param x0_ Physical coordinates of the start point.
        /// @param x1_ Physical coordinates of the end point.
        __host__ __device__ Edge(double2 x0_, double2 x1_)
        {
            x0 = x0_;
            dx = {x1_.x - x0_.x, x1_.y - x0_.y};
            const double s = sqrt(dx.x * dx.x + dx.y * dx.y);
            // Right-hand perpendicular: rotate (dx.x, dx.y) by 90° clockwise → (dx.y, -dx.x)
            n = {dx.y / s, -dx.x / s};
            meas = 0.5 * s;
        }

        /// @brief Maps reference coordinate xi in [-1,1] to physical coordinates.
        constexpr double2 physical_coordinates(double xi) const
        {
            const double t = 0.5 * (xi + 1.0);
            return {x0.x + dx.x * t, x0.y + dx.y * t};
        }

        /// @brief Returns the unit outward normal (constant along the edge).
        constexpr double2 normal() const { return n; }

        /// @brief Returns normal() * measure() — the weighted outward normal.
        constexpr double2 weighted_normal() const { return {n.x * meas, n.y * meas}; }

        /// @brief Returns the half-length of the edge (the edge measure weight).
        ///
        /// This is the magnitude of the Jacobian of the reference-to-physical map,
        /// i.e. |dx/dxi| = |x1 - x0| / 2, which is constant for a straight edge.
        constexpr double measure() const { return meas; }

        /// @brief Returns the total length of the edge.
        constexpr double length() const { return 2.0 * meas; }
    };

    /// @brief Topological connectivity of an edge in the mesh.
    struct EdgeConnectivity
    {
        int elements[2]; ///< global element indices (elements[1] == -1 for boundary)
        int labels[2];   ///< local side index {0..3} for each element
        int permutation; ///< +1 if DOF traversal order agrees between elements, -1 otherwise
    };

    constexpr int2 edge2vol(int N, int i, int label)
    {
        const int m = (label == 0 || label == 2) ? i : (label == 1) ? (N - 1) : 0;
        const int n = (label == 1 || label == 3) ? i : (label == 2) ? (N - 1) : 0;

        return {m, n};
    }

    constexpr int permute_edge_index(int N, int i, int permutation)
    {
        cuddh_assert(permutation == 1 || permutation == -1,
                     printf("permute_edge_index error: permutation must be one of {-1, 1}.\n"));
        return (permutation < 0) ? (N - 1 - i) : i;
    }
} // namespace cuddh

#endif
