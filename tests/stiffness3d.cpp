#include <cmath>

#include "test_common.hpp"

using namespace cuddh;

__device__ static double func(double3 r)
{
    const double x5 = pow(r.x, 5);
    const double y3 = pow(r.y, 3);
    const double z4 = pow(r.z, 4);
    return (x5 - 5.0 * r.x) * (y3 - 3.0 * r.y) + (z4 - 2.0 * r.z * r.z);
}

__device__ static double L(double3 r)
{
    const double x3 = pow(r.x, 3);
    const double x4 = r.x * x3;

    return 4.0 - 6.0 * r.x * r.y * (x4 - 5.0) - 20.0 * x3 * r.y * (r.y * r.y - 3.0) - 12.0 * r.z * r.z;
}

static void run_stiffness3d_case(TestLogger &summary, const Mesh3D &mesh, const Basis &basis,
                                 const std::string &test_name)
{
    constexpr double tol = 1e-6;

    H1Space3D fem(mesh, basis);
    const int ndof = fem.size();

    host_device_dvec _Af(ndof);
    host_device_dvec _f(ndof);
    host_device_dvec _Lf(ndof);

    double *Af = _Af.device_write();
    double *f = _f.device_write();
    double *Lf = _Lf.device_write();

    auto x = fem.physical_coordinates(MemorySpace::DEVICE);

    forall(ndof, [=] __device__(int i) -> void {
        const double3 xi = x[i];
        f[i] = func(xi);
    });

    MassMatrix3D M(fem);
    l2_project(M, [=] __device__(const double3 r) -> double { return L(r); }, Lf);

    StiffnessMatrix3D A(fem);

    if (dla::is_symmetric(ndof, A, 1e-10))
        summary.pass(std::format("stiffness3d {} is symmetric.", test_name));
    else
        summary.fail(std::format("stiffness3d {} is not symmetric.", test_name), "|x'Ay - y'Ax| > 1e-10.");

    A.action(f, Af);

    const double err = dla::dist(ndof, Af, Lf) / dla::norm(ndof, Lf);
    if (err < tol)
        summary.pass(std::format("stiffness3d {}", test_name));
    else
        summary.fail(std::format("stiffness3d {}", test_name),
                     std::format("relative error {} exceeds tolerance {}", err, tol));
}

static void run_stiffness3d_tests(TestLogger &summary)
{
    const int nx = 10;
    const Mesh3D mesh = Mesh3D::uniform_cube(nx, -1.0, 1.0, nx, -1.0, 1.0, nx, -1.0, 1.0);

    for (int p : {5, 6, 7, 8})
    {
        const Basis basis(p);
        run_stiffness3d_case(summary, mesh, basis, std::format("uniform cube p={}", p));
    }
}

int main()
{
    TestLogger summary;
    run_stiffness3d_tests(summary);
    return summary.finish();
}
