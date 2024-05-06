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
    const float gmres_tol = 1e-5; // relative tolerance. GMRES stops when ||b-A*x|| < tol*||b||
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
    mi.action(d_a, d_a);

    const double * h_a = a.host_read();
    DDH F(omega, h_a, fem, nx, nx);
    const int n_lambda = F.size();

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

    F.rhs(d_b, d_Y);

    auto out = gmres(n_lambda, d_L, &F, d_Y, gmres_m, gmres_maxit, gmres_tol, gmres_verbose);
    F.postprocess(d_L, d_b, d_U); // get solution from solution of substructured problem

    // copy solution to host
    const double * h_U = U.host_read();

    // save solution and collocation nodes to file
    auto xy = fem.physical_coordinates(MemorySpace::HOST);
    to_file("solution/xy.0000", N, xy);
    to_file("solution/ddh.0000", N, h_U);
}