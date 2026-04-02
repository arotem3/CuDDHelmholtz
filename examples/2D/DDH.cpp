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

    double r = (x + 0.5) * (x + 0.5) + y * y;
    double F = s / M_PI * std::exp(-s * r);

    r = (x - 0.5) * (x - 0.5) + (y + 0.5) * (y + 0.5);
    F += s / M_PI * std::exp(-s * r);
    return F;
}

// variable coefficient
__device__ static double alpha(const double X[2])
{
    const double r = X[0] * X[0] + X[1] * X[1];

    if (r < 0.0625)
        return 0.2;
    else
        return 1.0;
}

int main()
{
    const int deg = 3;                       // polynomial degree of basis functions
    const int nx = 64, ny = 64;              // number of elements along each direction. Mesh will have nx^2 elements
    const double omega = 2 * M_PI * nx / 10; // Helmholtz frequency

    const DDKernelConfig config = {
        .block_size = DDKernelConfig::Default, // one of Default, t256, t512, t1024
        .tdof = 2                              // one of 0, 1, 2, 3, 4.
    };

    const SolverParams opts = {
        .maxit = 1000,                       // maximum number of iterations of GMRES
        .rtol = 1e-5,                        // relative tolerance. GMRES stops when ||b-A*x|| < tol*||b||
        .verbose = SolverParams::ProgressBar // verbosity level: Silent, ProgressBar, Iteration
    };

    // Assemble the mesh
    Mesh2D mesh = Mesh2D::uniform_rect(nx, -1.0, 1.0, ny, -1.0, 1.0);

    // Construct 1D basis functions. On each element, the 2D basis functions are
    // tensor products of these 1D basis functions.
    Basis basis(deg + 1);

    // The mesh and 1D basis functions are combined in H1Space2D to define the
    // total global degrees of freedom of the problem.
    H1Space2D fem(mesh, basis);
    EnsembleSpace efem = partition_uniform_rect(fem, {nx, ny}, {8, 8});

    const int ndof = fem.size(); // # of degrees of freedom

    const int N = 2 * ndof; // total degrees of freedom in [u, v] (U := u + i v)

    auto ddh = [&]() {
        auto a = gridfunc(fem, [] __device__(const double X[2]) -> double { return alpha(X); });
        double *d_a = thrust::raw_pointer_cast(a.data());
        return DDH<float>(omega, d_a, fem, efem, config);
    }();

    thrust::universal_vector<double> U(N, 0.0);
    thrust::universal_vector<double> B(N, 0.0);

    double *u = thrust::raw_pointer_cast(U.data()); // the solution vector [u; v]
    double *b = thrust::raw_pointer_cast(B.data()); // the right hand side b(phi)

    l2_project(b, MassMatrix(fem), [=] __device__(const double X[2]) -> double {
        return f(X, omega);
    }); // compute the right hand side b(phi) = (f, phi)

    std::cout << "Solving the Helmholtz equation...\n"
              << "\tomega = " << omega << "\n"
              << "\t#elements = " << mesh.n_elem() << "\n"
              << "\tpolynomial degree = " << deg << "\n"
              << "\t#dof = " << 2 * ndof << "\n"
              << "\t#subdomains = " << efem.size() << "\n"
              << "\tmax #elements / subdomain = " << efem.max_n_elem() << "\n"
              << "\tmax #dof / subdomain = " << efem.max_size() << "\n"
              << "\tkernel = {" << ddh.op().kernel_str() << "}\n"
              << "\t#lambda = " << ddh.n_lambda() << std::endl;

    auto out = ddh.solve(u, b, opts);

    double res = [&]() -> double {
        ivec boundary_faces = mesh.boundary_edges();                 // identify boundary faces
        TraceSpace2D fs(fem, boundary_faces.size(), boundary_faces); // define trace space

        auto a2 = gridfunc(fem, [] __device__(const double X[2]) -> double {
            double ax = alpha(X);
            return ax * ax;
        });
        double *d_a2 = thrust::raw_pointer_cast(a2.data()); // variable coefficient projected onto H1Space2D

        auto a = trace(fs, [] __device__(const double X[2]) -> double { return alpha(X); });
        double *d_a = thrust::raw_pointer_cast(a.data()); // variable coefficient projected onto TraceSpace2D

        Helmholtz A(omega, d_a2, d_a, fem, fs);

        thrust::universal_vector<double> _Au(N, 0.0);
        auto Au = thrust::raw_pointer_cast(_Au.data());
        A.action(u, Au);

        return dla::dist(N, Au, b) / dla::norm(N, b);
    }();

    std::cout << std::format("Helmholtz residual |b - A u| / |b| ~ {:.2e}", res) << std::endl;

    CUDDH_CUDA_CHECK(cudaDeviceSynchronize());

    // save solution and collocation nodes to file
    auto xy = fem.physical_coordinates(MemorySpace::HOST);

    const char xy_file[] = "solution/xy.0000";
    const char sol_file[] = "solution/ddh.0000";
    const char res_file[] = "solution/residuals.0000";

    if (to_file(xy_file, N, xy.data()))
        std::cout << "\ncoordinates written to: " << xy_file << "\n";
    if (to_file(sol_file, N, u))
        std::cout << "Solution written to: " << sol_file << "\n";
    if (to_file(res_file, out.res_norm.size(), out.res_norm.data()))
        std::cout << "Residuals written to: " << res_file << "\n";

    return 0;
}