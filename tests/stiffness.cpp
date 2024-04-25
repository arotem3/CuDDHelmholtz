#include "test.hpp"
#include <fstream>
#include <iomanip>

/// a function with zero normal derivative on boundary of [-1,1]x[-1,1] 
__device__ static double func(const double X[2])
{
    const double x = X[0], y = X[1];
    double x5 = std::pow(x, 5);
    double y3 = std::pow(y, 3);
    return (x5 - 5.0 * x) * (y3 - 3.0 * y);
}

// negative laplacian of func
__device__ static double L(const double X[2])
{
    const double x = X[0], y = X[1];
    double x3 = std::pow(x, 3);
    double x5 = std::pow(x, 5);
    double y3 = std::pow(y, 3);
    return -6.0*y*(x5-5*x) - 20.0*x3*(y3-3.0*y);
}

using namespace cuddh;

namespace cuddh_test
{
    void t_stiffness(int& n_test, int& n_passed, const Mesh2D& mesh, const Basis& basis, const QuadratureRule& quad, const std::string& test_name)
    {
        constexpr double tol = 1e-6;
        const int n_basis = basis.size();

        H1Space fem(mesh, basis);
        const int ndof = fem.size();

        host_device_dvec _Af(ndof);
        host_device_dvec _f(ndof);
        host_device_dvec _Lf(ndof);
        
        double * Af = _Af.device_write();
        double * f = _f.device_write();
        double * Lf = _Lf.device_write();

        auto x = fem.physical_coordinates(MemorySpace::DEVICE);

        // evaluate func
        forall(ndof, [=] __device__ (int i) -> void
        {
            const double xi[] = {x(0, i), x(1, i)};
            f[i] = func(xi);
        });

        // (L, phi)
        LinearFunctional l(fem, quad);
        l.action([=] __device__ (const double X[2]) -> double {return L(X);}, Lf);

        StiffnessMatrix A(fem, quad);
        A.action(f, Af);

        const double err = dist(ndof, Af, Lf) / cuddh::norm(ndof, Lf);

        n_test++;
        if (err < tol)
        {
            std::cout << "\t[ + ] t_stiffness(" << test_name << ") test successful." << std::endl;
            n_passed++;
        }
        else
        {
            std::cout << "\t[ - ] t_stiffness(" << test_name << ") test failed.\n\t\tComputed error ~ " << err << "> tol (" << tol << ")." << std::endl;
        }
    }

    void t_stiffness(int& n_test, int& n_passed)
    {
        {
            const int nx = 10;
            Mesh2D mesh = Mesh2D::uniform_rect(nx, -1.0, 1.0, nx, -1.0, 1.0);
            for (int p : {6, 7, 8})
            {
                Basis basis(p);
                QuadratureRule q(p+2, QuadratureRule::GaussLegendre);
                std::string test_name = "structured mesh | p = " + std::to_string(p);
                t_stiffness(n_test, n_passed, mesh, basis, q, test_name);
            }
        }

        {
            Mesh2D mesh = load_unstructured_square();
            for (int p : {6, 7, 8})
            {
                Basis basis(p);
                QuadratureRule q(p+2, QuadratureRule::GaussLegendre);
                std::string test_name = "unstructured mesh | p = " + std::to_string(p);
                t_stiffness(n_test, n_passed, mesh, basis, q, test_name);
            }
        }
    }
} // namespace cuddh_test
