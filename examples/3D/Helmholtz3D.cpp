/**
 * @file Helmholtz3D.cpp
 * @brief Example driver for solving the 3D Helmholtz equation
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
 * In cuddh, a is assembled from the following operators:
 *
 *      StiffnessMatrix3D S;
 *      S.action(c, x, y); // y[i] <- y[i] + c * (grad x, grad phi[i])
 *
 *      MassMatrix3D M;
 *      M.action(c, x, y); // y[i] <- y[i] + c * (a(x)^2 x, phi[i])
 *
 *      FaceMassMatrix3D H;
 *      H.action(c, x, y); // y[i] <- y[i] + c * <a(x) x, phi[i]>
 *
 * The Helmholtz3D class combines these operations to define the bilinear form a(*,*).
 *
 * To compile & run this program:
 *  (1) From the CuDDHelmholtz directory, compile the library:
 *      cmake .
 *      make cuddh -j
 *  (2) compile the program:
 *      make Helmholtz3D
 *  (3) run:
 *      ./examples/Helmholtz3D
 *
 * The program will write the collocation points to `solution/coo.0000` in binary
 * format. The solution is written to `solution/helmholtz.0000` in binary
 * format.
 *
 * This format can be read and visualized, for example, in Python using numpy and matplotlib via:
 *
 *      coo = numpy.fromfile("solution/coo.0000").view(numpy.float64).reshape(-1, 3)
 *      x, y, z = coo[:, 0], coo[:, 1], coo[:, 2]
 *
 *      uv = numpy.fromfile("solution/helmholtz.0000", dtype=numpy.float64).reshape(-1, 2)
 *      U  = uv[:, 0] + 1j * uv[:, 1]
 */

#include "cuddh.hpp"
#include "examples.hpp"

using namespace cuddh;

/// @brief forcing term, approximate point source
__device__ static double f(double3 x, double omega)
{
    double s = omega * omega;

    double r1 = (x.x - 0.5) * (x.x - 0.5) + x.y * x.y + x.z * x.z;
    double F = std::pow(s / M_PI, 1.5) * std::exp(-s * r1);

    double r2 = (x.x + 0.7) * (x.x + 0.7) + (x.y + 0.7) * (x.y + 0.7) + x.z * x.z;
    F += std::pow(s / M_PI, 1.5) * std::exp(-s * r2);

    return F;
}

/// @brief a(x) = 1/c(x) where c(x) is the wave-speed.
__device__ static double a(double3 x)
{
    const double r = x.x * x.x + x.y * x.y + x.z * x.z;
    return (r < 0.0625) ? 0.2 : 1.0;
}

int main()
{
    const int deg = 2;                       // polynomial degree of basis functions
    const int nx = 32;                       // number of elements along each direction
    const double omega = 2 * M_PI * nx / 10; // Helmholtz frequency

    const SolverParams opts = {
        .maxit = 100'000,                    // maximum number of iterations of MINRES
        .rtol = 1e-5,                        // relative tolerance
        .verbose = SolverParams::ProgressBar // verbosity level
    };

    // Assemble the mesh
    Mesh3D mesh = Mesh3D::uniform_cube(nx, -1.0, 1.0, nx, -1.0, 1.0, nx, -1.0, 1.0);

    // Construct 1D basis functions
    Basis basis(deg + 1);

    // Combine the mesh and basis functions in H1Space3D to define the global DOFs
    H1Space3D fem(mesh, basis);
    const int ndof = fem.size();

    std::cout << "Solving the 3D Helmholtz equation...\n"
              << "\tomega = " << omega << "\n"
              << "\t#elements = " << mesh.n_elem() << "\n"
              << "\tpolynomial degree = " << deg << "\n"
              << "\t#dof = " << 2 * ndof << "\n";

    // Identify boundary faces and construct the trace space
    auto boundary_faces = mesh.get_boundary_faces();
    TraceSpace3D fs(fem, boundary_faces.size(), boundary_faces);

    host_device_dvec a2x(ndof);
    host_device_dvec ax(fs.size());

    gridfunc(
        fem,
        [=] __device__(double3 x) -> double {
            double aX = a(x);
            return aX * aX;
        },
        a2x.device_write());

    trace(fs, [=] __device__(double3 x) -> double { return a(x); }, ax.device_write());

    Helmholtz3D A(omega, a2x.device_read(), ax.device_read(), fem, fs);

    const int N = 2 * ndof;

    thrust::universal_vector<double> U(N, 0.0); // solution vector [u; v] initialized to zero
    thrust::universal_vector<double> B(N, 0.0); // right-hand side [b; 0]

    double *u = thrust::raw_pointer_cast(U.data());
    double *b = thrust::raw_pointer_cast(B.data());

    MassMatrix3D M(fem);
    l2_project(M, [=] __device__(double3 x) -> double { return f(x, omega); }, b);

    // solve a([u, v], phi) = b(phi)
    std::cout << "\nsolving with MINRES ... \n";
    auto out = minres(N, u, A, b, opts);
    CUDDH_CUDA_CHECK(cudaDeviceSynchronize());

    // save solution and collocation nodes to file
    auto coo = fem.physical_coordinates(MemorySpace::HOST);

    const char coo_file[] = "solution/coo.0000";
    const char sol_file[] = "solution/helmholtz.0000";
    const char res_file[] = "solution/residuals.0000";

    if (to_file(coo_file, coo.size(), coo.data()))
        std::cout << "Saved collocation points to " << coo_file << std::endl;
    else
        std::cerr << "Failed to save collocation points to " << coo_file << std::endl;

    if (to_file(sol_file, N, u))
        std::cout << "Saved solution to " << sol_file << std::endl;
    else
        std::cerr << "Failed to save solution to " << sol_file << std::endl;

    if (to_file(res_file, out.res_norm.size(), out.res_norm.data()))
        std::cout << "Saved residuals to " << res_file << std::endl;
    else
        std::cerr << "Failed to save residuals to " << res_file << std::endl;

    return 0;
}
