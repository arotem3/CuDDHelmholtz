#ifndef CUDDH_ELEMENT_HPP
#define CUDDH_ELEMENT_HPP

#include "cuddh_config.hpp"
#include "Tensor.hpp"

namespace cuddh
{
    /// @brief Abstract representation of a finite element.
    class Element
    {
    public:
        int id; ///< @brief global index of element in the mesh.
                ///<
                ///< The existing mesh constructors always assign this value, and
                ///< it is used by some functions.
        int nodes[4];

        /// @brief maps the reference coordinates `xi` to the physical coordinates `x`.
        ///
        /// @param[in] xi coordinates in the reference element. Shape (2,)
        /// @param[out] x on exit, coordinates in physical space. Shape (2,)
        virtual void physical_coordinates(const double * xi, double * x) const = 0;
        
        /// @brief computes the Jacobian of the element mapping from reference
        /// coordinates to physical coordinates.
        /// 
        /// The Jacobian is defined as 
        /// $$J_{ij} = \frac{\partial x_j}{\partial \xi_i},$$
        /// so that \f$\nabla_x = J^{-1}\nabla_{\xi}\f$ where \f$x\f$ is the physical
        /// coordinate and \f$\xi\f$ is the reference coordinate. We define the
        /// Jacobian this way because it is more natural to compute
        /// \f$\frac{\partial x_j}{\partial \xi_i}\f$ than
        /// \f$\frac{\partial\xi_j}{\partial x_i}\f$. However,
        /// \f$\frac{\partial\xi_j}{\partial x_i}\f$ can be computed by inverting
        /// this \f$2\times 2\f$ matrix.
        ///
        /// @param[in] xi coordinates in the reference element. Shape (2,)
        /// @param[out] J on exit, Jacobian matrix. Shape (2, 2)
        virtual void jacobian(const double * xi, double * J) const = 0;

        /// @brief computes the determinant of the jacobian of element mapping
        /// from reference coordinates to physical coordinates.
        ///
        /// The output is returns \f$\mu(\xi) := \det(J)\f$ the determinant of the
        /// Jacobian of element mapping from reference coordinates to physical
        /// coordinates. Then \f$dx = \mu(\xi) d\xi\f$. If this function is not
        /// overloaded, then `measure` will call `jacobian` and return the
        /// determinant of the result.
        ///
        /// @param[in] xi coordinates in the reference element. Shape (2,)
        /// @return The measure weight at `xi`
        virtual double measure(const double * xi) const
        {
            double J[4];
            jacobian(xi, J);
            return J[0]*J[3] - J[1]*J[2];
        }
    
        virtual double area() const = 0;

        virtual ~Element() = default;
    };

    /// @brief The `QuadElement` is a straight sides quadrilateral element. It
    /// is defined by the coordinates of its four corners. The reference element
    /// for this type is \f$[-1, 1]^2\f$.
    class QuadElement : public Element
    {
    private:
        double x[4][2];

    public:
        /// @brief maps the reference coordinates `xi` to the physical coordinates `x`.
        ///
        /// @param[in] xi coordinates in the reference element. Shape (2,)
        /// @param[out] x on exit, coordinates in physical space. Shape (2,)
        void physical_coordinates(const double * xi, double * x) const override;

        /// @brief computes the Jacobian of the element mapping from reference
        /// coordinates to physical coordinates.
        /// 
        /// The Jacobian is defined as 
        /// $$J_{ij} = \frac{\partial x_j}{\partial \xi_i},$$
        /// so that \f$\nabla_x = J^{-1}\nabla_{\xi}\f$ where \f$x\f$ is the physical
        /// coordinate and \f$\xi\f$ is the reference coordinate. We define the
        /// Jacobian this way because it is more natural to compute
        /// \f$\frac{\partial x_j}{\partial \xi_i}\f$ than
        /// \f$\frac{\partial\xi_j}{\partial x_i}\f$. However,
        /// \f$\frac{\partial\xi_j}{\partial x_i}\f$ can be computed by inverting
        /// this \f$2\times 2\f$ matrix.
        ///
        /// @param[in] xi coordinates in the reference element. Shape (2,)
        /// @param[out] J on exit, Jacobian matrix. Shape (2, 2)
        void jacobian(const double * xi, double * J) const override;

        /// @brief returns the area of the element 
        double area() const override;

        /// @brief initialize `QuadElement` by providing the coordinates of its
        /// corners (in counter clockwise order).
        /// 
        /// The input array `X` should have shape (2, 4) so that \f$X_{0,i} = x_i\f$
        /// and \f$X_{1,i} = y_i\f$ for \f$i=1,...,4\f$. The input `X` is copied.
        /// @param[in] X physical coordinates of element corners. Shape (2, 4).
        QuadElement(const double * X);

        ~QuadElement() = default;

        /// @brief returns a pointer to the coordinates of the `i`-th corner of
        /// the element as ordered on construction.
        ///
        /// @param[in] i index of corner
        /// @return A pointer to the physical coordinates. They are an array of
        /// length 2. It should not be assumed that `corner(i+1) = corner(i) +
        /// 2`.
        inline const double * corner(int i) const
        {
            return x[i];
        }
    };
} // namespace dg

#endif