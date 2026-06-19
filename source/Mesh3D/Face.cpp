#include "Mesh3D/Face.hpp"

using namespace cuddh;

QuadFace::QuadFace(const double3 *X)
{
    for (int i = 0; i < 4; ++i)
        x[i] = X[i];
}

double3 QuadFace::physical_coordinates(const double2 &s) const
{
    const double b[] = {0.25 * (1.0 - s.x) * (1.0 - s.y),
                        0.25 * (1.0 + s.x) * (1.0 - s.y),
                        0.25 * (1.0 + s.x) * (1.0 + s.y),
                        0.25 * (1.0 - s.x) * (1.0 + s.y)};

    double3 coo = {0.0, 0.0, 0.0};

    for (int i = 0; i < 4; ++i)
    {
        coo.x += b[i] * x[i].x;
        coo.y += b[i] * x[i].y;
        coo.z += b[i] * x[i].z;
    }

    return coo;
}

double3x2 QuadFace::jacobian(const double2 &s) const
{
    double3x2 J;

    // Compute partial derivatives of physical coordinates with respect to s
    double dx[] = {-0.25 * (1.0 - s.y), 0.25 * (1.0 - s.y), 0.25 * (1.0 + s.y), -0.25 * (1.0 + s.y)};
    double dy[] = {-0.25 * (1.0 - s.x), -0.25 * (1.0 + s.x), 0.25 * (1.0 + s.x), 0.25 * (1.0 - s.x)};

    // Now calculate the Jacobian matrix entries
    J(0, 0) = dx[0] * x[0].x + dx[1] * x[1].x + dx[2] * x[2].x + dx[3] * x[3].x;
    J(1, 0) = dx[0] * x[0].y + dx[1] * x[1].y + dx[2] * x[2].y + dx[3] * x[3].y;
    J(2, 0) = dx[0] * x[0].z + dx[1] * x[1].z + dx[2] * x[2].z + dx[3] * x[3].z;

    J(0, 1) = dy[0] * x[0].x + dy[1] * x[1].x + dy[2] * x[2].x + dy[3] * x[3].x;
    J(1, 1) = dy[0] * x[0].y + dy[1] * x[1].y + dy[2] * x[2].y + dy[3] * x[3].y;
    J(2, 1) = dy[0] * x[0].z + dy[1] * x[1].z + dy[2] * x[2].z + dy[3] * x[3].z;

    return J;
}

double3 QuadFace::weighted_normal(const double2 &s) const
{
    double3x2 J = jacobian(s);

    double3 n = {J(1, 0) * J(2, 1) - J(2, 0) * J(1, 1),
                 J(2, 0) * J(0, 1) - J(0, 0) * J(2, 1),
                 J(0, 0) * J(1, 1) - J(1, 0) * J(0, 1)};

    return n;
}

double3 QuadFace::normal(const double2 &s) const
{
    double3 n = weighted_normal(s);
    double norm = std::sqrt(n.x * n.x + n.y * n.y + n.z * n.z);

    n.x /= norm;
    n.y /= norm;
    n.z /= norm;

    return n;
}

double QuadFace::measure(const double2 &s) const
{
    double3 n = weighted_normal(s);
    return std::sqrt(n.x * n.x + n.y * n.y + n.z * n.z);
}
