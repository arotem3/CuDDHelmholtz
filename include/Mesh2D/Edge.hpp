#ifndef CUDDH_EDGE_HPP
#define CUDDH_EDGE_HPP

#include <cmath>

#include "cuddh_config.hpp"

namespace cuddh
{
    enum class FaceType
    {
        INTERIOR, ///< @brief Edge is on interior of mesh.
                  ///< 
                  ///< Edges on interior have elements on both sides, thus it
                  ///< is expected that `elements[1]` and `sides[1]` are defined.
        BOUNDARY  ///< @brief Edge is on boundary of mesh.
                  ///<
                  ///< Edges on the boundary have only one elemented
                  ///< connected, thus `element[1]` and `sides[1]` need not be
                  ///< specified, so referencing these values will result in
                  ///< undefined behavior.
    };

    /// @brief generic representation of an edge of a finite element.
    struct Edge
    {
    public:
        FaceType type; ///< whether the variable is on the interior or boundary.
        int id; ///< @brief global index of edge in the mesh.
                ///<
                ///< The existing mesh constructors always assign this value,
                ///< and it is used by some functions. The member `id` refers to
                ///< the global index rather than the local index, and is not
                ///< modified when a mesh is distributed.

        int nodes[2]; ///< @brief The global index of the nodes on this edge
        int elements[2]; ///< @brief The global index of the elements on this edge.
                         ///<
                         ///< `elements[0]` is always specified, and
                         ///< `elements[1]` is defined only if `type == INTERIOR`.

        int sides[2]; ///< @brief Identifies the edge relative to the element, it is one of {0, 1, 2, 3}.
                      ///<
                      ///< `sides[0]` is always specified, and `sides[1]` is
                      ///< defined only if `type == INTERIOR`.
        int delta; ///< defines the direction in which to iterate through degrees of
                   ///< freedom on the edge for the second element on the edge. This value is
                   ///< either +1 or -1 depending on the relative orientation of the two
                   ///< elements.
    
        /// @brief assigns to `n` the unit normal to the edge in the outward
        /// direction from the first element on the edge evaluated at the
        /// reference coordinate `xi` in the reference interval [-1, 1].
        /// @param[in] xi coordinates in the reference interval. Shape (2,)
        /// @param[out] n unit normal at xi. Shape (2,)
        virtual void normal(const double xi, double * n) const = 0;

        /// Returns \f$\mu(\xi) = \frac{d}{d\xi}(\mathbf{n}\cdot x)\f$ where
        /// \f$x\f$ is the physical coordinate, \f$\xi\in[-1,1]\f$ is the reference
        /// coordinate, and \f$\mathbf{n}\f$ is the unit normal to the edge. Then
        /// \f$\mathbf{n}\cdot dx = \mu(\xi) d\xi\f$ on the edge.
        /// @param[in] xi coordinates in the reference interval. Shape (2,)
        /// @return edge measure.
        virtual double measure(const double xi) const = 0;

        /// @brief assigns to `x` the physical coordinate on the edge at point
        /// `xi` in the reference interval [-1, 1].
        /// @param xi coordinates in the reference interval. Shape (2,)
        /// @param x coordinates in physical space. Shape (2,)
        virtual void physical_coordinates(const double xi, double * x) const = 0;

        virtual double length() const = 0;

        Edge() : id{-1}, elements{-1, -1}, sides{-1, -1} {}
        virtual ~Edge() = default;
    };

    /// @brief The `SraightEdge` is a line segment. It is defined by the
    /// coordinates of its two end points. It maps the reference interval
    /// \f$[-1, 1]\f$ to the line segment.
    struct StraightEdge : public Edge
    {
    private:
        double n[2];
        double meas;
        double x[2];
        double dx[2];

    public:
        /// @brief constructs a `StraightEdge` by specifying the coordinates of
        /// its endpoints and the side identifier of the first element on this
        /// edge which is needed to determine the sign of the normal vector. The
        /// coordinates are copied. `this->sides[0]` is initialized with `side`.
        /// @param x0 physical coordinates of start point. Shape (2,)
        /// @param x1 physical coordinates of end point. Shape (2,)
        /// @param side The side indentifier of this edge with respect to the
        /// first element on this edge.
        StraightEdge(const double * x0, const double * x1, int side)
        {
            x[0] = x0[0];
            x[1] = x0[1];

            dx[0] = x1[0] - x0[0];
            dx[1] = x1[1] - x0[1];

            const double s = std::hypot(dx[0], dx[1]);
            const double sgn = (side == 2 || side == 3) ? -1 : 1;

            n[0] = sgn * dx[1] / s;
            n[1] = -sgn * dx[0] / s;

            meas = s / 2;
        }

        ~StraightEdge() = default;

        /// @brief assigns to `n` the unit normal to the edge in the outward
        /// direction from the first element on the edge evaluated at the
        /// reference coordinate `xi` in the reference interval [-1, 1].
        /// @param[in] xi coordinates in the reference interval. Shape (2,)
        /// @param[out] n unit normal at xi. Shape (2,)
        inline void normal(const double xi, double * n_) const override
        {
            n_[0] = n[0];
            n_[1] = n[1];
        }

        /// Returns \f$\mu(\xi) = \frac{d}{d\xi}(\mathbf{n}\cdot x)\f$ where
        /// \f$x\f$ is the physical coordinate, \f$\xi\in[-1,1]\f$ is the reference
        /// coordinate, and \f$\mathbf{n}\f$ is the unit normal to the edge. Then
        /// \f$\mathbf{n}\cdot dx = \mu(\xi) d\xi\f$ on the edge.
        /// @param[in] xi coordinates in the reference interval. Shape (2,)
        /// @return edge measure.
        inline double measure(const double) const override
        {
            return meas;
        }
    
        /// @brief assigns to `x` the physical coordinate on the edge at point
        /// `xi` in the reference interval [-1, 1].
        /// @param xi coordinates in the reference interval. Shape (2,)
        /// @param x coordinates in physical space. Shape (2,)
        inline void physical_coordinates(const double xi, double * x_) const override
        {
            const double t = 0.5 * (xi + 1.0);

            x_[0] = x[0] + dx[0] * t;
            x_[1] = x[1] + dx[1] * t;
        }
    
        /// @brief returns the length of edge 
        inline double length() const override
        {
            // meas = sqrt(dx^2 + dy^2)/2 so length = 2*meas
            return 2.0 * meas;
        }
    };
} // namespace dg


#endif