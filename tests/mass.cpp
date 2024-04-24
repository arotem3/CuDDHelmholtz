#include "test.hpp"

 __host__ __device__ static double func(const double X[2])
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
        host_device_dvec _b(ndof);
        host_device_dvec _Mf(ndof);
        
        double * u = _u.device_write(); // projected func
        double * f = _f.device_write(); // interpolated values of func
        double * b = _b.device_write(); // (f, phi)
        double * Mf = _Mf.device_write(); // mass matrix times f

        auto X = fem.physical_coordinates(MemorySpace::DEVICE);

        // evaluate f on nodes
        forall(ndof, [=] __device__ (int i) -> void
        {
            double xy[2];
            xy[0] = X(0, i);
            xy[1] = X(1, i);
            f[i] = func(xy);
        });
        
        // evaluate (f, phi)
        LinearFunctional l(fem, quad);
        l.action(1.0, [] __device__ (double X[2]) {return func(X);}, b);

        MassMatrix m(fem);
        DiagInvMassMatrix p(fem);

        m.action(f, Mf);

        double err = dist(ndof, Mf, b) / cuddh::norm(ndof, b);

        n_test++;
        if (err > 1e-8)
            std::cout << "\tt_mass(): test \"" << test_name << "\" failed forward problem with error ~ " << err << "\n";
        else
            n_passed++;
        
        // solve for u
        const int gmres_m = 5;
        const int maxiter = 10;
        const double tol = 1e-12;
        auto out = gmres(ndof, u, &m, b, &p, gmres_m, maxiter, tol);

        err = dist(ndof, u, f) / cuddh::norm(ndof, f);

        n_test++;
        if (err > 1e-8)
            std::cout << "\tt_mass(): test \"" << test_name << "\" failed inverse problem with error ~ " << err << "\n";
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
