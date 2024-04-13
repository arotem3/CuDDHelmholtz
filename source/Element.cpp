#include "Element.hpp"

namespace cuddh
{
    void QuadElement::physical_coordinates(const double * xi, double * x_) const
    {
        const double b[] = {0.25 * (1.0 - xi[0]) * (1.0 - xi[1]),
                            0.25 * (1.0 + xi[0]) * (1.0 - xi[1]),
                            0.25 * (1.0 + xi[0]) * (1.0 + xi[1]),
                            0.25 * (1.0 - xi[0]) * (1.0 + xi[1])};
        x_[0] = 0.0;
        x_[1] = 0.0;

        for (int i=0; i < 4; ++i)
        {
            x_[0] += x[i][0] * b[i];
            x_[1] += x[i][1] * b[i];
        }
    }

    void QuadElement::jacobian(const double * xi, double * J) const
    {
        J[0] = 0.25 * ((1.0 - xi[1]) * (x[1][0] - x[0][0]) + (1.0 + xi[1]) * (x[2][0] - x[3][0])); // dx/d(xi)
        J[1] = 0.25 * ((1.0 - xi[1]) * (x[1][1] - x[0][1]) + (1.0 + xi[1]) * (x[2][1] - x[3][1])); // dy/d(xi)
        J[2] = 0.25 * ((1.0 - xi[0]) * (x[3][0] - x[0][0]) + (1.0 + xi[0]) * (x[2][0] - x[1][0])); // dx/d(eta)
        J[3] = 0.25 * ((1.0 - xi[0]) * (x[3][1] - x[0][1]) + (1.0 + xi[0]) * (x[2][1] - x[1][1])); // dy/d(eta)
    }

    double QuadElement::area() const
    {
        // det(Jacobian) is linear, so it is integrated exactly by Gauss
        // quadrature with a single point (x=0, w=2, so tensor product ->
        // (x,y)=(0,0), w=4)
        double zero[] = {0.0, 0.0};
        return 4.0 * measure(zero);
    }

    QuadElement::QuadElement(const double * xs)
    {
        auto X = reshape(xs, 2, 4);

        for (int i=0; i < 4; ++i)
        {
            x[i][0] = X(0, i);
            x[i][1] = X(1, i);
        }
    }
    
} // namespace dg
