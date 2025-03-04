#include "Mesh3D/Element.hpp"

using namespace cuddh;

static constexpr double shape_function(int i, const double3 &s)
{
    double x = (i == 0 || i == 3 || i == 4 || i == 7) ? -1.0 : 1.0;
    double y = (i == 0 || i == 1 || i == 4 || i == 5) ? -1.0 : 1.0;
    double z = (i < 4) ? -1.0 : 1.0;

    return 0.125 * (1.0 + x * s.x) * (1.0 + y * s.y) * (1.0 + z * s.z);
}

static constexpr double3 shape_function_gradient(int i, const double3 &s)
{
    double x = (i == 0 || i == 3 || i == 4 || i == 7) ? -1.0 : 1.0;
    double y = (i == 0 || i == 1 || i == 4 || i == 5) ? -1.0 : 1.0;
    double z = (i < 4) ? -1.0 : 1.0;

    return double3{
        0.125 * x * (1.0 + y * s.y) * (1.0 + z * s.z),
        0.125 * y * (1.0 + x * s.x) * (1.0 + z * s.z),
        0.125 * z * (1.0 + x * s.x) * (1.0 + y * s.y)};
}

HexElement::vec HexElement::physical_coordinates(const vec &s) const
{
    vec coo{0.0, 0.0, 0.0};

    for (int i = 0; i < 8; i++)
    {
        double p = shape_function(i, s);

        coo.x += p * x[i].x;
        coo.y += p * x[i].y;
        coo.z += p * x[i].z;
    }

    return coo;
}

HexElement::mat HexElement::jacobian(const vec &s) const
{
    mat J;
    zeros(J);

    for (int i = 0; i < 8; i++)
    {
        vec grad = shape_function_gradient(i, s);

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

double HexElement::measure(const vec &s) const
{
    mat J = jacobian(s);
    return det(J);
}

HexElement::HexElement(const vec *X)
{
    for (int i = 0; i < 8; i++)
        x[i] = X[i];
}