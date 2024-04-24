#include "test.hpp"
#include <fstream>
#include <iomanip>

/// a function with zero normal derivative on boundary of [-1,1]x[-1,1] 
static double __host__ __device__ func(const double X[2])
{
    const double x = X[0], y = X[1];
    double x5 = std::pow(x, 5);
    double y3 = std::pow(y, 3);
    return (x5 - 5.0 * x) * (y3 - 3.0 * y);
}

// negative laplacian of func
static double __host__ __device__ L(const double X[2])
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
        n_test++;

        const int n_basis = basis.size();

        H1Space fem(mesh, basis);
        const int ndof = fem.size();

        host_device_dvec _u(ndof);
        host_device_dvec _Au(ndof);
        host_device_dvec _f(ndof);
        host_device_dvec _Lf(ndof);
        
        double * u = _u.device_write();
        double * Au = _Au.device_write();
        double * f = _f.device_write();
        double * Lf = _Lf.device_write();

        LinearFunctional l(fem, quad);
        l.action(1.0, func, f);
        l.action(1.0, L, Lf);

        MassMatrix m(fem);
        DiagInvMassMatrix p(fem);
        auto out = gmres(ndof, u, &m, f, &p, 20, 10, 1e-12); // (u, v) == (f, v) for all v
        if (not out.success)
        {
            std::cout << "\tt_stiffness(): test \"" << test_name << "\" failed... something wrong with the mass matrix?\n";
            return;
        }

        StiffnessMatrix A(fem, quad);
        A.action(1.0, u, Au);

        const double * h_Au = _Au.host_read();
        const double * h_Lf = _Lf.host_read();

        double max_err = 0.0;
        for (int i = 0; i < ndof; ++i)
        {
            double err = h_Au[i] - h_Lf[i];
            max_err = std::max(std::abs(err), max_err);
        }

        if (max_err > 1e-6)
            std::cout << "\tt_stiffness(): test \"" << test_name << "\" failed with error " << max_err << "\n";
        else
            n_passed++;
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
