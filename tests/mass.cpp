#include "test.hpp"

static double __host__ __device__ func(const double X[2])
{
    const double x = X[0], y = X[1];
    return 3.0 * x * x - 2.0 * x * y + y + 1.0;
}

using namespace cuddh;

namespace cuddh_test
{
    void t_mass(int& n_test, int& n_passed, const Mesh2D& mesh, const Basis& basis, const QuadratureRule& quad, const std::string& test_name)
    {
        const int n_elem = mesh.n_elem();
        const int n_basis = basis.size();

        H1Space fem(mesh, basis);
        const int ndof = fem.size();

        host_device_dvec _u(ndof);
        host_device_dvec _f(ndof);
        
        double * u = _u.device_write();
        double * f = _f.device_write();

        LinearFunctional l(fem, quad);
        l.action(1.0, func, f);

        MassMatrix m(fem);
        DiagInvMassMatrix p(fem);

        const int gmres_m = 20;
        const int maxiter = 10;
        const double tol = 1e-12;
        auto out = gmres(ndof, u, &m, f, &p, gmres_m, maxiter, tol);

        // verify correctness
        const double * h_u = _u.host_read();
        const double * h_f = _f.host_read();

        auto I = fem.global_indices(MemorySpace::HOST);
        auto x = fem.physical_coordinates(MemorySpace::HOST);

        double max_err = 0.0;
        for (int i = 0; i < ndof; ++i)
        {
            double xi[2];
            xi[0] = x(0, i);
            xi[1] = x(1, i);

            double fi = func(xi);
            double err = h_u[i] - fi;
            max_err = std::max(std::abs(err), max_err);
        }

        n_test++;
        if (max_err > 1e-8)
            std::cout << "\tt_mass(): test \"" << test_name << "\" failed with error " << max_err << "\n";
        else
            n_passed++;
    }

    void t_mass(int& n_test, int& n_passed)
    {
        {
            const int nx = 10;
            Mesh2D mesh = Mesh2D::uniform_rect(nx, -1.0, 1.0, nx, -1.0, 1.0);
            for (int p : {3, 4, 5, 6, 7, 8})
            {
                Basis basis(p);
                QuadratureRule quad(p+2, QuadratureRule::GaussLegendre);
                std::string test_name = "structured mesh | p = " + std::to_string(p);

                t_mass(n_test, n_passed, mesh, basis, quad, test_name);
            }
        }

        {
            Mesh2D mesh = load_unstructured_square();
            for (int p : {3, 4, 5, 6, 7, 8})
            {
                Basis basis(p);
                QuadratureRule quad(p+2, QuadratureRule::GaussLegendre);
                std::string test_name = "unstructured mesh | p = " + std::to_string(p); 

                t_mass(n_test, n_passed, mesh, basis, quad, test_name);
            }
        }
    }
} // namespace cuddh
