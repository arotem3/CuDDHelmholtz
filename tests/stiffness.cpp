#include <cmath>
#include <random>

#include "test_common.hpp"

using namespace cuddh;

// Function with zero normal derivative on the boundary of [-1,1]x[-1,1].
__device__ static double func(double2 X)
{
    const auto [x, y] = X;
    const double x5 = std::pow(x, 5);
    const double y3 = std::pow(y, 3);
    return (x5 - 5.0 * x) * (y3 - 3.0 * y);
}

__device__ static double L(double2 X)
{
    const auto [x, y] = X;
    const double x3 = std::pow(x, 3);
    const double x5 = std::pow(x, 5);
    const double y3 = std::pow(y, 3);
    return -6.0 * y * (x5 - 5 * x) - 20.0 * x3 * (y3 - 3.0 * y);
}

static void accuracy_test(TestLogger &summary, const Mesh2D &mesh, const Basis &basis, const std::string &test_name)
{
    constexpr double tol = 1e-6;

    H1Space2D fem(mesh, basis);
    const int ndof = fem.size();

    host_device_dvec _Af(ndof);
    host_device_dvec _Lf(ndof);

    double *Af = _Af.device_write();
    double *Lf = _Lf.device_write();

    auto _f = gridfunc(fem, [=] __device__(double2 x) { return func(x); });
    double *f = thrust::raw_pointer_cast(_f.data());

    l2_project(Lf, MassMatrix(fem), [=] __device__(double2 x) { return L(x); });

    StiffnessMatrix A(fem);
    A.action(f, Af);

    const double err = dla::dist(ndof, Af, Lf) / dla::norm(ndof, Lf);
    if (err < tol)
        summary.pass(std::format("stiffness {}", test_name));
    else
        summary.fail(std::format("stiffness {}", test_name),
                     std::format("relative error {} exceeds tolerance {}", err, tol));
}

static void symmetry_test(TestLogger &summary, const Mesh2D &mesh, const Basis &basis, std::string test_name)
{
    constexpr double tol = 1e-10;

    H1Space2D fem(mesh, basis);
    const int ndof = fem.size();

    StiffnessMatrix A(fem);

    if (dla::is_symmetric(A, tol))
        summary.pass(std::format("stiffness {} symmetry test", test_name));
    else
        summary.fail(std::format("stiffness {} symmetry test", test_name),
                     std::format("|x'Ay - y'Ax| = exceeds tolerance {}", tol));
}

static void run_stiffness_tests(TestLogger &summary)
{
    {
        const int nx = 10;
        Mesh2D mesh = Mesh2D::uniform_rect(nx, -1.0, 1.0, nx, -1.0, 1.0);
        for (int p : {6, 7, 8})
        {
            Basis basis(p);
            std::string name = std::format("structured mesh p={}", p);
            accuracy_test(summary, mesh, basis, name);
            symmetry_test(summary, mesh, basis, name);
        }
    }

    {
        Mesh2D mesh = load_unstructured_square();
        for (int p : {6, 7, 8})
        {
            Basis basis(p);
            std::string name = std::format("unstructured mesh p={}", p);
            accuracy_test(summary, mesh, basis, name);
            symmetry_test(summary, mesh, basis, name);
        }
    }
}

int main()
{
    TestLogger summary;
    run_stiffness_tests(summary);
    return summary.finish();
}
