#include "test.hpp"

static double func(const double X[2])
{
    const double x = X[0], y = X[1];
    return 3.0 * x * x - 2.0 * x * y + y + 1.0;
}

using namespace cuddh;

class mass_op : public Operator
{
public:
    mass_op(const H1Space& fem) : n{fem.size()}, m(fem) {}

    void action(const double * x, double * y) const override
    {
        for (int i = 0; i < n; ++i)
            y[i] = 0.0;

        m.action(1.0, x, y);
    }

private:
    const int n;
    MassMatrix m;
};

class mass_prec : public Operator
{
public:
    mass_prec(const H1Space& fem) : n{fem.size()}, p(fem) {}

    void action(const double * x, double * y) const override
    {
        for (int i = 0; i < n; ++i)
            y[i] = 0.0;
        p.action(1.0, x, y);
    }

private:
    const int n;
    DiagInvMassMatrix p;
};

namespace cuddh_test
{
    void t_mass(int& n_test, int& n_passed, const Mesh2D& mesh, const Basis& basis, const QuadratureRule& quad, const std::string& test_name)
    {
        const int n_elem = mesh.n_elem();
        const int n_basis = basis.size();

        H1Space fem(mesh, basis);
        const int ndof = fem.size();

        dvec u(ndof);
        dvec f(ndof);

        LinearFunctional l(fem, quad);
        l.action(func, f);

        mass_op m(fem);
        mass_prec p(fem);

        const int gmres_m = 20;
        const int maxiter = 10;
        const double tol = 1e-12;
        auto out = gmres(ndof, u, &m, f, &p, gmres_m, maxiter, tol);

        // verify correctness
        auto I = fem.global_indices();
        auto x = reshape(mesh.element_metrics(basis.quadrature()).physical_coordinates(), 2, n_basis, n_basis, n_elem);

        double max_err = 0.0;
        for (int el = 0; el < n_elem; ++el)
        {
            for (int j = 0; j < n_basis; ++j)
            {
                for (int i = 0; i < n_basis; ++i)
                {
                    double xij[2];
                    xij[0] = x(0, i, j, el);
                    xij[1] = x(1, i, j, el);

                    double fij = func(xij);
                    double err = u[I(i, j, el)] - fij;
                    max_err = std::max(std::abs(err), max_err);
                }
            }
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
