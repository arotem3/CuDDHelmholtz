#include <cmath>

#include "test_common.hpp"

using namespace cuddh;

// Function with zero normal derivative on the boundary of [-1,1]x[-1,1].
__device__ static double func(const double X[2])
{
    const double x = X[0], y = X[1];
    const double x5 = std::pow(x, 5);
    const double y3 = std::pow(y, 3);
    return (x5 - 5.0 * x) * (y3 - 3.0 * y);
}

__device__ static double L(const double X[2])
{
    const double x = X[0], y = X[1];
    const double x3 = std::pow(x, 3);
    const double x5 = std::pow(x, 5);
    const double y3 = std::pow(y, 3);
    return -6.0 * y * (x5 - 5 * x) - 20.0 * x3 * (y3 - 3.0 * y);
}

static void run_stiffness_case(TestLogger &summary, const Mesh2D &mesh, const Basis &basis, const QuadratureRule &quad,
                               const std::string &test_name)
{
    constexpr double tol = 1e-6;

    H1Space2D fem(mesh, basis);
    const int ndof = fem.size();

    host_device_dvec _Af(ndof);
    host_device_dvec _f(ndof);
    host_device_dvec _Lf(ndof);

    double *Af = _Af.device_write();
    double *f = _f.device_write();
    double *Lf = _Lf.device_write();

    auto x = fem.physical_coordinates(MemorySpace::DEVICE);

    forall(ndof, [=] __device__(int i) -> void {
        const double xi[] = {x(0, i), x(1, i)};
        f[i] = func(xi);
    });

    LinearFunctional l(fem, quad);
    l.action([=] __device__(const double X[2]) -> double { return L(X); }, Lf);

    StiffnessMatrix A(fem, quad);
    A.action(f, Af);

    const double err = dist(ndof, Af, Lf) / cuddh::norm(ndof, Lf);
    if (err < tol)
        summary.pass(std::format("stiffness {}", test_name));
    else
        summary.fail(std::format("stiffness {}", test_name),
                     std::format("relative error {} exceeds tolerance {}", err, tol));
}

static void run_stiffness_tests(TestLogger &summary)
{
    {
        const int nx = 10;
        Mesh2D mesh = Mesh2D::uniform_rect(nx, -1.0, 1.0, nx, -1.0, 1.0);
        for (int p : {6, 7, 8})
        {
            Basis basis(p);
            QuadratureRule q(p + 2, QuadratureRule::GaussLegendre);
            run_stiffness_case(summary, mesh, basis, q, std::format("structured mesh p={}", p));
        }
    }

    {
        Mesh2D mesh = load_unstructured_square();
        for (int p : {6, 7, 8})
        {
            Basis basis(p);
            QuadratureRule q(p + 2, QuadratureRule::GaussLegendre);
            run_stiffness_case(summary, mesh, basis, q, std::format("unstructured mesh p={}", p));
        }
    }
}

int main()
{
    TestLogger summary;
    run_stiffness_tests(summary);
    return summary.finish();
}
