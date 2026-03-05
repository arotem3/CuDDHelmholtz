/**
 * @file Helmholtz.cpp
 * @brief Example driver for solving the Helmholtz equation
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
 * In cuddh, a is assembled from the following operators:
 *
 *      StiffnessMatrix S;
 *      S.action(c, x, y); // y[i] <- y[i] + c * (grad x, grad phi[i]) where phi[i] is the i-th basis function
 *
 *      MassMatrix M;
 *      M.action(c, x, y); // y[i] <- y[i] + c * (a(x)^2 x, phi[i])
 *
 *      FaceMassMatrix H;
 *      H.action(c, x, y); // y[i] <- y[i] + c * <a(x)  x, phi[i]>
 *
 * The Helmholtz class combines these operations to define the bilinear form a(*,*).
 *
 * To compile & run this program:
 *  (1) From the CuDDHelmholtz directory, compile the library:
 *      cmake .
 *      make cuddh -j
 *  (2) compile the program:
 *      make Helmholtz
 *  (3) run:
 *      ./examples/Helmholtz
 *
 * The program will write the collocation points to `solution/xy.0000` in binary
 * format. The solution is written to `solution/helmholtz.0000` in binary
 * format.
 *
 * This format can be read and visualized, for example, in Python using numpy and matplotlib via:
 *
 *      xy = numpy.fromfile("solution/xy.0000", order='F')
 *      xy = xy.reshape(2, -1)
 *      x, y = xy[0], xy[1]
 *
 *      uv = numpy.fromfile("solution/helmholtz.0000", order='F')
 *      uv = uv.reshape(-1, 2)
 *      U  = uv[:, 0] + 1j * uv[:, 1]
 *
 *      # visualize the modulus of U
 *      matplotlib.pyplot.tricontourf(x, y, np.abs(U))
 */

#include "Helmholtz.hpp"

#include "cuddh.hpp"
#include "examples.hpp"

using namespace cuddh;

/// @brief forcing term, approximate point source
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

/// @brief a(x) = 1/c(x) where c(x) is the wave-speed.
__device__ static double a(const double X[2])
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
    const int nx = 64;                       // number of elements along each direction. Mesh will have nx^2 elements
    const double omega = 2 * M_PI * nx / 10; // Helmholtz frequency

    const SolverParams opts = {
        .maxit = 100'000,                    // maximum number of iterations of MINRES
        .tol = 1e-6,                         // relative tolerance. MINRES stops when ||b-A*x|| < tol*||b||
        .verbose = SolverParams::ProgressBar // verbosity level: ProgressBar, Iteration, or Silent
    };

    // Assemble the mesh
    Mesh2D mesh = Mesh2D::uniform_rect(nx, -1.0, 1.0, nx, -1.0, 1.0);

    // Construct 1D basis functions. On each element, the 2D basis functions are
    // tensor products of these 1D basis functions.
    Basis basis(deg + 1);

    // The mesh and 1D basis functions are combined in H1Space2D to define the
    // total global degrees of freedom of the problem.
    H1Space2D fem(mesh, basis);
    const int ndof = fem.size(); // # of degrees of freedom

    std::cout << "Solving the Helmholtz equation...\n"
              << "\tomega = " << omega << "\n"
              << "\t#elements = " << mesh.n_elem() << "\n"
              << "\tpolynomial degree = " << deg << "\n"
              << "\t#dof = " << 2 * ndof << "\n";

    auto A = [&]() -> Helmholtz {
        // identify the boundary faces in the mesh in order to define the TraceSpace2D
        // and FaceMassMatrix
        ivec boundary_faces = mesh.boundary_edges();

        // The TraceSpace2D is a subspace of the H1Space2D used to identify the degrees
        // of freedom needed in the computation of trace terms: <u, phi>
        TraceSpace2D fs(fem, boundary_faces.size(), boundary_faces);

        auto a2x = gridfunc(fem, [] __device__(const double X[2]) -> double {
            double aX = a(X);
            return aX * aX;
        });

        auto ax = trace(fs, [] __device__(const double X[2]) -> double { return a(X); });

        double *d_a2 = thrust::raw_pointer_cast(a2x.data()); // a^2(x) projected onto H1Space2D
        double *d_a = thrust::raw_pointer_cast(ax.data());   // a(x) projected onto TraceSpace2D

        return Helmholtz(omega, d_a2, d_a, fem, fs);
    }();

    const int N = 2 * ndof; // total degrees of freedom in [u, v] (U := u + i v)

    thrust::universal_vector<double> U(N, 0.0); // solution vector [u; v] initialized to zero
    thrust::universal_vector<double> B(N, 0.0);

    double *u = thrust::raw_pointer_cast(U.data()); // the solution vector [u; v]
    double *b = thrust::raw_pointer_cast(B.data()); // the right hand side b(phi)

    l2_project(b, MassMatrix(fem), [=] __device__(const double X[2]) -> double { return f(X, omega); });

    // solve a([u, v], phi) = b(phi)
    std::cout << "\nsolving with MINRES ... \n";
    auto out = minres(N, u, &A, b, opts);

    // save solution and collocation nodes to file
    auto xy = fem.physical_coordinates(MemorySpace::HOST);

    const char xy_file[] = "solution/xy.0000";
    const char sol_file[] = "solution/helmholtz.0000";
    const char res_file[] = "solution/residuals.0000";

    if (to_file(xy_file, 2 * ndof, xy.data()))
        std::cout << "Saved collocation points to " << xy_file << std::endl;
    else
        std::cerr << "Failed to save collocation points to " << xy_file << std::endl;

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
