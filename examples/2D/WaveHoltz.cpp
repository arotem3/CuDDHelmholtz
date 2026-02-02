/**
 * @file WaveHoltz.cpp
 * @brief Example driver for solving the Helmholtz equation with the WaveHoltz solver.
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
 * The WaveHoltz class implements the finite element discretization of this equation,
 * but is actually solving the wave eqution at each iteration.
 * 
 * To compile & run this program:
 *   (1) From the build directory, compile the library:
 *      cmake ..
 *      make cuddh -j
 *  (2) compile the program:
 *     make WaveHoltz
 *  (3) run:
 *     ./examples/WaveHoltz
 * 
 * The program will write the collocation points to `solution/xy.0000` in binary
 * format. The solution is written to `solution/waveholtz.0000` in binary
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
#include "Helmholtz.hpp"
#include <format>

using namespace cuddh;

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

__device__ static double alpha(const double X[2])
{
    const double r = X[0] * X[0] + X[1] * X[1];

    if (r < 0.0625)
        return 0.2;
    else
        return 1.0;
}

constexpr double maxvel = 5.0; // maximum reciprocal of alpha

int main()
{
    const int deg = 3; // polynomial degree
    const int nx = 128; // number of elements in each direction
    const double omega = 2 * M_PI * nx / 10; // Helmholtz frequency

    const int gmres_m = 20; // GMRES restart parameter
    const int gmres_maxit = 100; // maximum number of GMRES iterations
    const double gmres_tol = 1e-6; // GMRES tolerance
    const int gmres_verbose = 1; // GMRES verbosity level

    // Create a uniform rectangular mesh
    Mesh2D mesh = Mesh2D::uniform_rect(nx, -1.0, 1.0, nx, -1.0, 1.0);

    // Construct 1D basis functions
    Basis basis(deg + 1);

    // Combine the mesh and basis functions in H1Space2D to define the total global degrees of freedom
    H1Space2D fem(mesh, basis);

    const int ndof = fem.size(); // number of degrees of freedom
    const int N = 2 * ndof; // total degrees of freedom in [u, v] (U := u + i v)

    ivec boundary_faces = mesh.boundary_edges(); // identify boundary faces
    TraceSpace2D fs(fem, boundary_faces.size(), boundary_faces); // define trace space

    const int fdof = fs.size(); // number of degrees of freedom in trace space

    host_device_dvec U(N); // solution vector
    host_device_dvec b(N); // right-hand side vector
    host_device_dvec Gb(N); // WaveHoltz G applied to b
    host_device_dvec a2(ndof); /// variable coefficient on collocation points
    host_device_dvec a(fdof); // variable coefficient on trace space

    double *d_U = U.device_write(); // device pointer to solution vector
    double *d_b = b.device_write(); // device pointer to right-hand side vector
    double *d_a2 = a2.device_write(); // device pointer to variable coefficient
    double *d_a = a.device_write(); // device pointer to variable coefficient on trace space
    double *d_Gb = Gb.device_write(); // device pointer to G applied to b

    LinearFunctional l(fem);
    DiagInvMassMatrix mi(fem);
    l.action([=] __device__(const double X[2]) -> double { return f(X, omega); }, d_b); // compute right-hand side
    
    l.action([=] __device__(const double X[2]) -> double { double ax = alpha(X); return ax*ax; }, d_a2); // compute variable coefficient
    mi.action(d_a2, d_a2); // project the coefficient onto the basis

    FaceLinearFunctional fl(fs);
    DiagInvFaceMassMatrix fmi(fs);

    fl.action([=] __device__(const double X[2]) -> double { return alpha(X); }, d_a); // compute variable coefficient on trace space
    fmi.action(d_a, d_a); // project the coefficient onto the trace space

    // Initialize the WaveHoltz operator
    WaveHoltz W(omega, maxvel, d_a2, d_a, fem, fs);

    W.G(d_b, d_Gb); // apply G to the right-hand side vector

    std::cout << "Solving the Helmholtz equation...\n"
              << "\tomega = " << omega << "\n"
              << "\t#elements = " << mesh.n_elem() << "\n"
              << "\tpolynomial degree = " << deg << "\n"
              << "\t#dof = " << 2 * ndof << std::endl;

    // Solve the system using GMRES
    solver_out out = gmres(N, d_U, &W, d_Gb, gmres_m, gmres_maxit, gmres_tol, gmres_verbose);

    double res_norm = [&]()
    {
        host_device_dvec res(N);
        double *d_res = res.device_write();

        Helmholtz A(omega, d_a2, d_a, fem, fs);
        A.action(d_U, d_res); // compute residuals
        axpby(N, -1.0, d_b, 1.0, d_res); // res = A U - b
        return cuddh::norm(N, d_res) / cuddh::norm(N, d_b); // compute norm of residual
    }();

    std::cout << "Relative residual norm ||A U - b|| / ||b|| = " << res_norm << std::endl;

    // save the solution to a file
    auto xy = fem.physical_coordinates(MemorySpace::HOST);

    auto xyfile = "solution/xy.0000";
    auto solfile = "solution/waveholtz.0000";
    auto resfile = "solution/residuals.0000";

    if (to_file(xyfile, 2 * ndof, xy.data()))
        std::cout << "Saved collocation points to " << xyfile << std::endl;
    else
        std::cerr << "Failed to save collocation points to " << xyfile << std::endl;

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