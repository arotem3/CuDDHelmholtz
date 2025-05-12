/**
 * @file WaveHoltz.cpp
 * @brief Example driver for solving the Helmholtz equation with the WaveHoltz solver.
 * 
 * @details This file is a driver for solving the Helmholtz equation with
 * approximate absorbing boundary conditions:
 * 
 *      -div(grad U) - omega^2 a^2(x) U == f    in  D := [-1, 1]^3
 *      i a(x) omega U + dU/dn == 0             on boundary of D
 *
 * Here omega is the frequency. We assume f is real valued, and U is complex
 * valued.
 * 
 * The WaveHoltz3D class implements the finite element discretization of this equation,
 * but is actually solving the wave eqution at each iteration.
 *
 * The program will write the collocation points to `solution/coo.0000` in binary
 * format. The solution is written to `solution/waveholtz.0000` in binary
 * format. The residuals from the GMRES iterations are written to `solution/residuals.0000`
 * in binary format.
 */

#include "cuddh.hpp"
#include "examples.hpp"
#include <format>

using namespace cuddh;

__device__ static double f(double3 x, double omega)
{
    double s = omega * omega;
    double r1 = (x.x-0.5)*(x.x-0.5) + x.y * x.y + x.z * x.z;
    double F1 = std::pow(s / M_PI, 1.5) * std::exp(-s * r1);

    double r2 = (x.x+0.2)*(x.x+0.2) + (x.y-0.7)*(x.y-0.7) + (x.z-0.3)*(x.z-0.3);
    double F2 = std::pow(s / M_PI, 1.5) * std::exp(-s * r2);

    return F1 + F2;
}

__device__ static double alpha(double3 x)
{
    const double r = x.x * x.x + x.y * x.y + x.z * x.z;
    return (r < 0.0625) ? 0.2 : 1.0;
}

constexpr double maxvel = 5.0; // maximum reciprocal of alpha

int main()
{
    const int deg = 2; // polynomial degree
    const int nx = 64; // number of elements in x direction
    const double omega = 2 * M_PI * nx / 10; // time-harmonic frequency

    const int gmres_m = 20; // GMRES restart parameter
    const int gmres_maxit = 100; // maximum number of GMRES iterations
    const double gmres_tol = 1e-6; // GMRES tolerance
    const int gmres_verbose = 1; // GMRES verbosity level

    // Create a uniform rectangular mesh
    Mesh3D mesh = Mesh3D::uniform_cube(nx, -1.0, 1.0, nx, -1.0, 1.0, nx, -1.0, 1.0);

    // Construct 1D basis functions
    Basis basis(deg + 1);

    // Combine the mesh and basis functions in H1Space3D to define the total global degrees of freedom
    H1Space3D fem(mesh, basis);

    const int ndof = fem.size(); // number of degrees of freedom
    const int N = 2 * ndof; // total degrees of freedom in [u, v] (U := u + i v)

    auto boundary_faces = mesh.get_boundary_faces(); // identify boundary faces
    TraceSpace3D fs(fem, boundary_faces.size(), boundary_faces); // define trace space

    const int fdof = fs.size(); // number of degrees of freedom in trace space

    host_device_dvec U(N); // solution vector
    host_device_dvec b(N); // forcing term
    host_device_dvec Gb(N); // WaveHoltz G applied to b
    host_device_dvec a2(ndof); // variable coefficient on collocation points
    host_device_dvec a(fdof); // variable coefficient on trace space

    double *d_U = U.device_write(); // device pointer to solution vector
    double *d_b = b.device_write(); // device pointer to right-hand side vector
    double *d_a2 = a2.device_write(); // device pointer to variable coefficient
    double *d_a = a.device_write(); // device pointer to variable coefficient on trace space
    double *d_Gb = Gb.device_write(); // device pointer to G applied to b

    // compute b = (f, phi)
    MassMatrix3D M(fem);
    l2_project(M, [=] __device__(double3 x) -> double { return f(x, omega); }, d_b); // compute right-hand side

    // compute variable coefficient
    gridfunc(fem, [=] __device__(double3 x) -> double { double ax = alpha(x); return ax*ax; }, d_a2); // compute variable coefficient

    // compute trace of the variable coefficient
    trace(fs, [=] __device__(double3 x) -> double { return alpha(x); }, d_a);

    // Initialize the WaveHoltz operator
    WaveHoltz3D W(omega, maxvel, d_a2, d_a, fem, fs);

    W.G(d_b, d_Gb); // apply G to the right-hand side vector

    // Peek at the last CUDA error without resetting it
    cudaError_t err = cudaPeekAtLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA peek error after W.G: " << cudaGetErrorString(err) << std::endl;
    }

    std::cout << "Solving the Helmholtz equation...\n"
              << "\tomega = " << omega << "\n"
              << "\t#elements = " << mesh.n_elem() << "\n"
              << "\tpolynomial degree = " << deg << "\n"
              << "\t#dof = " << 2 * ndof << std::endl;

    // Solve the system using GMRES
    solver_out out = gmres(N, d_U, &W, d_Gb, gmres_m, gmres_maxit, gmres_tol, gmres_verbose);

    // save the solution to a file
    auto coo = fem.physical_coordinates(MemorySpace::HOST);

    auto coofile = "solution/coo.0000";
    auto solfile = "solution/waveholtz.0000";
    auto resfile = "solution/residuals.0000";

    if (to_file(coofile, ndof, coo.data()))
        std::cout << "Saved collocation points to " << coofile << std::endl;
    else
        std::cerr << "Failed to save collocation points to " << coofile << std::endl;

    if (to_file(solfile, N, U.host_read()))
        std::cout << "Saved solution to " << solfile << std::endl;
    else
        std::cerr << "Failed to save solution to " << solfile << std::endl;

    if (to_file(resfile, out.res_norm.size(), out.res_norm.data()))
        std::cout << "Saved residuals to " << resfile << std::endl;
    else
        std::cerr << "Failed to save residuals to " << resfile << std::endl;

    return 0;
}