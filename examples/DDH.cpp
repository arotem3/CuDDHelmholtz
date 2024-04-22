#include "cuddh.hpp"
#include "examples.hpp"

using namespace cuddh;

static double f(const double X[2])
{
    const double x = X[0] + 0.5, y = X[1];
    const double r = x * x + y * y;
    return 20.0 * std::exp(-400.0 * r);
}

int main()
{
    const double omega = 10.0; // Helmholtz frequency
    const int deg = 3; // polynomial degree of basis functions
    const int nx = 32; // number of elements along each direction. Mesh will have nx^2 elements
    
    // Assemble the mesh
    Mesh2D mesh = Mesh2D::uniform_rect(nx, -1.0, 1.0, nx, -1.0, 1.0);

    // Construct 1D basis functions. On each element, the 2D basis functions are
    // tensor products of these 1D basis functions.
    Basis basis(deg+1);

    // The mesh and 1D basis functions are combined in H1Space to define the
    // total global degrees of freedom of the problem.
    H1Space fem(mesh, basis);
    const int ndof = fem.size(); // # of degrees of freedom

    std::cout << "Solving the Helmholtz equation...\n"
              << "\tomega = " << omega << "\n"
              << "\t#elements = " << mesh.n_elem() << "\n"
              << "\tpolynomial degree = " << deg << "\n"
              << "\t#dof = " << ndof << "\n";
    
    const int N = 2 * ndof; // total degrees of freedom in [u, v] (U := u + i v)

    dvec U(N); // the solution vector [u; v]
    dvec b(N); // the right hand side b(phi)

    LinearFunctional l(fem); // computes projections like (f, phi)
    l.action(1.0, f, b); // bu[i] <- (f, phi[i])

    DDH op(omega, fem, nx, nx);

    op.action(b, U);

    // save solution and collocation nodes to file
    auto xy = fem.physical_coordinates();
    to_file("solution/xy.00000", N, xy);
    to_file("solution/ddh.00000", N, U);
}