#include "test.hpp"

__device__ static double func(double3 r)
{
    double x5 = pow(r.x, 5);
    double y3 = pow(r.y, 3);
    double z4 = pow(r.z, 4);
    return (x5 - 5.0 * r.x) * (y3 - 3.0 * r.y) + (z4 - 2.0 * r.z * r.z);
}

__device__ static double L(double3 r)
{
    double x3 = pow(r.x, 3);
    double x4 = r.x * x3;

    return 4.0 - 6.0 * r.x * r.y * (x4 - 5.0) - 20.0 * x3 * r.y * (r.y * r.y - 3.0) - 12.0 * r.z * r.z;
}

using namespace cuddh;

static void t_stiffness3d(int &n_test, int &n_passed, const Mesh3D &mesh, const Basis &basis, const std::string &test_name)
{
    constexpr double tol = 1e-6;
    const int n_basis = basis.size();

    H1Space3D fem(mesh, basis);
    const int ndof = fem.size();

    host_device_dvec _Af(ndof);
    host_device_dvec _f(ndof);
    host_device_dvec _Lf(ndof);

    double *Af = _Af.device_write();
    double *f = _f.device_write();
    double *Lf = _Lf.device_write();

    auto x = fem.physical_coordinates(MemorySpace::DEVICE);

    // evaluate func
    forall(ndof, [=] __device__(int i) -> void
    {
        const double3 xi = x[i];
        f[i] = func(xi);
    });

    MassMatrix3D M(fem);
    l2_project(M, [=] __device__ (const double3 r) -> double { return L(r); }, Lf);

    StiffnessMatrix3D A(fem);
    A.action(f, Af);

    const double err = dist(ndof, Af, Lf) / cuddh::norm(ndof, Lf);

    if (err < tol)
    {
        std::cout << "\t[ + ] t_stiffness3d("  << test_name << ") test successful." << std::endl;
        n_passed++;
    }
    else
    {
        std::cout << "\t[ - ] t_stiffness3d(" << test_name << ") test failed.\n\t\tComputed error ~ " << err << " > tol (" << tol << ")" << std::endl;
    }

    n_test++;
}

void cuddh_test::t_stiffness3d(int &n_test, int &n_passed)
{
    const int nx = 10;
    const Mesh3D mesh = Mesh3D::uniform_cube(nx, -1.0, 1.0, nx, -1.0, 1.0, nx, -1.0, 1.0);
    
    for (int p : {5, 6, 7, 8})
    {
        const Basis basis(p);
        std::string test_name = "uniform cube | p = " + std::to_string(p);

        ::t_stiffness3d(n_test, n_passed, mesh, basis, test_name);
    }
}
