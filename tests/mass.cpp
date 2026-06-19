#include "test_common.hpp"

__device__ static double func(double2 x)
{
    return 3.0 * x.x * x.x - 2.0 * x.x * x.y + x.y + 1.0;
}

using namespace cuddh;

static void run_mass_case(TestLogger &summary, const Mesh2D &mesh, Basis basis, std::string test_name)
{
    constexpr double tol = 1e-8;

    H1Space2D fem(mesh, basis);
    const int ndof = fem.size();

    host_device_dvec _u(ndof);
    host_device_dvec _b(ndof);
    host_device_dvec _Mf(ndof);

    double *u = _u.device_write();
    double *b = _b.device_write();
    double *Mf = _Mf.device_write();

    auto X = fem.physical_coordinates(MemorySpace::DEVICE);

    host_device_dvec _f(ndof);
    double *f = _f.device_write();

    // evaluate f at the global FEM degrees of freedom
    forall(ndof, [=] __device__(int i) mutable { f[i] = func(X(i)); });

    // evaluate (f, phi)
    MassMatrix m(fem);

    l2_project(b, m, [=] __device__(double2 x) { return func(x); });

    m.action(f, Mf);

    double err = dla::dist(ndof, Mf, b) / dla::norm(ndof, b);

    if (err < tol)
        summary.pass(std::format("mass forward {}", test_name));
    else
        summary.fail(std::format("mass forward {}", test_name),
                     std::format("relative error {} exceeds tolerance {}", err, tol));
}

static void run_mass_tests(TestLogger &summary)
{
    {
        const int nx = 10;
        Mesh2D mesh = Mesh2D::uniform_rect(nx, -1.0, 1.0, nx, -1.0, 1.0);
        for (int p : {3, 4, 5, 6, 7, 8})
            run_mass_case(summary, mesh, Basis(p), std::format("structured mesh p={}", p));
    }

    {
        Mesh2D mesh = load_unstructured_square();
        for (int p : {3, 4, 5, 6, 7, 8})
            run_mass_case(summary, mesh, Basis(p), std::format("unstructured mesh p={}", p));
    }
}

int main()
{
    TestLogger summary;
    run_mass_tests(summary);
    return summary.finish();
}
