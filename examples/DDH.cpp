/**
 * @file DDH.cpp
 * @brief Example driver for solving the Helmholtz equation with the domain decomposition solver
 *
 * @details This file is a driver for solving the Helmholtz equation with
 * approximate absorbing boundary conditions:
 *
 *      -div(grad U) - omega^2 a^2(x) U == f    in  D := [-1, 1]x[-1, 1]
 *      i a(x) omega U + dU/dn == 0             on boundary of D
 *
 * Here omega is the frequency. We assume f is real valued, and U is complex
 * valued.
 *
 * Write U = u + i v, the weak formulation is
 *
 *      a([u, v], phi) == b(phi)        for all phi in H1(D)
 *
 * The bilinear form a is defined as
 *
 *      a([u, v], phi) = [ (grad u, grad phi) - omega^2 (a^2(x) u, phi) - omega <a(x) v, phi>;
 *                         (grad v, grad phi) - omega^2 (a^2(x) v, phi) + omega <a(x) u, phi> ]
 *
 * And the linear operator b is defined b(phi) = [ (f, phi); 0 ]
 * 
 * The DDH class implements this discretization but is used to solve the 
 * substructured problem instead of the original problem.
 * 
 * To compile & run this program:
 *  (1) From the CuDDHelmholtz directory, compile the library:
 *      cmake .
 *      make cuddh -j
 *  (2) compile the program:
 *      make DDH
 *  (3) run:
 *      ./examples/DDH
 *
 * The program will write the collocation points to `solution/xy.0000` in binary
 * format. The solution is written to `solution/ddh.0000` in binary
 * format.
 *
 * This format can be read and visualized, for example, in Python using numpy and matplotlib via:
 *      
 *      xy = numpy.fromfile("solution/xy.0000", order='F')
 *      xy = xy.reshape(2, -1)
 *      x, y = xy[0], xy[1]
 *
 *      uv = numpy.fromfile("solution/ddh.0000", order='F')
 *      uv = uv.reshape(-1, 2)
 *      U  = uv[:, 0] + 1j * uv[:, 1]
 * 
 *      # visualize the modulus of U
 *      matplotlib.pyplot.tricontourf(x, y, np.abs(U))
 */

#include "cuddh.hpp"
#include "examples.hpp"

using namespace cuddh;

// forcing
__device__ static double f(const double X[2], double omega)
{
    const double x = X[0], y = X[1];
    double s = omega * omega;
    
    double r = (x+0.5)*(x+0.5) + y * y;
    double F = s / M_PI * std::exp(-s * r);
    
    r = (x-0.5)*(x-0.5) + (y+0.5)*(y+0.5);
    F += s / M_PI * std::exp(-s * r);
    return F;
}

// variable coefficient
__device__ static double alpha(const double X[2])
{
    const double r = X[0]*X[0] + X[1]*X[1];
    
    if (r < 0.0625)
        return 0.2;
    else
        return 1.0;
}

int main()
{
    const int deg = 3; // polynomial degree of basis functions
    const int nx = 128; // number of elements along each direction. Mesh will have nx^2 elements
    const double omega = 2 * M_PI * nx / 10; // Helmholtz frequency

    const int gmres_m = 20; // number of vectors in the Krylov space used in each iteration of GMRES
    const int gmres_maxit = 100; // maximum number of iterations of GMRES
    const float gmres_tol = 1e-4; // relative tolerance. GMRES stops when ||b-A*x|| < tol*||b||
    const int gmres_verbose = 1; // 0: silent, 1: progress bar, 2: one line per iteration
    
    // Assemble the mesh
    Mesh2D mesh = Mesh2D::uniform_rect(nx, -1.0, 1.0, nx, -1.0, 1.0);

    // Construct 1D basis functions. On each element, the 2D basis functions are
    // tensor products of these 1D basis functions.
    Basis basis(deg+1);

    // The mesh and 1D basis functions are combined in H1Space to define the
    // total global degrees of freedom of the problem.
    H1Space fem(mesh, basis);
    const int ndof = fem.size(); // # of degrees of freedom
    
    const int N = 2 * ndof; // total degrees of freedom in [u, v] (U := u + i v)

    host_device_dvec U(N);
    host_device_dvec b(N);
    host_device_dvec a(ndof);

    double * d_U = U.device_write(); // the solution vector [u; v]
    double * d_b = b.device_write(); // the right hand side b(phi)
    double * d_a = a.device_write(); // the variable coefficient

    LinearFunctional l(fem); // computes (f, phi)
    DiagInvMassMatrix mi(fem);
    l.action([=] __device__ (const double X[2]) -> double {return f(X, omega);}, d_b); // bu[i] <- (f, phi[i])

    l.action([] __device__ (const double X[2]) -> double {return alpha(X);}, d_a);
    mi.action(d_a, d_a); // project the coefficient d_a onto the basis

    const double * h_a = a.host_read(); // right now the DDH setup requires a host array for the coefficient.
    DDH F(omega, h_a, fem, nx, nx);
    const int n_lambda = F.size(); // number of degrees of freedom in substructured problem

    std::cout << "Solving the Helmholtz equation...\n"
              << "\tomega = " << omega << "\n"
              << "\t#elements = " << mesh.n_elem() << "\n"
              << "\tpolynomial degree = " << deg << "\n"
              << "\t#dof = " << 2 * ndof << "\n"
              << "\t#lambda = " << n_lambda << "\n";

    HostDeviceArray<float> L(n_lambda);
    HostDeviceArray<float> Y(n_lambda);
    float * d_L = L.device_write();
    float * d_Y = Y.device_write();

    F.rhs(d_b, d_Y); // compute the right hand side of the substructured problem from the Helmholtz right hand side

    auto out = gmres(n_lambda, d_L, &F, d_Y, gmres_m, gmres_maxit, gmres_tol, gmres_verbose);
    F.postprocess(d_L, d_b, d_U); // get solution from solution of substructured problem

    // copy solution to host
    const double * h_U = U.host_read();

    // save solution and collocation nodes to file
    auto xy = fem.physical_coordinates(MemorySpace::HOST);

    const char xy_file[] = "solution/xy.0000";
    const char sol_file[] = "solution/ddh.0000";
    to_file(xy_file, N, xy);
    to_file(sol_file, N, h_U);

    std::cout << "\nSolution written to: " << sol_file
              << "\nCoordinates written to: " << xy_file << "\n";

    return 0;
}