#include "cuddh.hpp"
#include "examples.hpp"

using namespace cuddh;

__device__ static double f(const double X[2])
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
              << "\t#dof = " << 2 * ndof << "\n";
    
    const int N = 2 * ndof; // total degrees of freedom in [u, v] (U := u + i v)

    host_device_dvec U(N);
    host_device_dvec b(N);

    double * d_U = U.device_write(); // the solution vector [u; v]
    double * d_b = b.device_write(); // the right hand side b(phi)

    LinearFunctional l(fem); // computes projections like (f, phi)
    l.action([] __device__ (const double X[2]) -> double {return f(X);}, d_b); // bu[i] <- (f, phi[i])

    DDH op(omega, fem, nx, nx);
    
    op.action(d_b, d_U);

    // copy solution to host
    const double * h_U = U.host_read();

    // save solution and collocation nodes to file
    auto xy = fem.physical_coordinates(MemorySpace::HOST);
    to_file("solution/xy.0000", N, xy);
    to_file("solution/ddh.0000", N, h_U);
}