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

#include <format>

#include "Helmholtz.hpp"
#include "cuddh.hpp"
#include "examples.hpp"

using namespace cuddh;

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
    const int deg = 3;                       // polynomial degree
    const int nx = 64;                       // number of elements in each direction
    const double omega = 2 * M_PI * nx / 10; // Helmholtz frequency

    const gmresParams opts = {
        .m = 50,                                 // GMRES restart parameter
        .maxit = 200,                            // maximum number of GMRES iterations
        .tol = 1e-6,                             // GMRES tolerance
        .verbose = SolverVerbosity::ProgressBar, // verbosity level: ProgressBar, Iteration, or Silent
    };

    // Create a uniform rectangular mesh
    Mesh2D mesh = Mesh2D::uniform_rect(nx, -1.0, 1.0, nx, -1.0, 1.0);

    // Construct 1D basis functions
    Basis basis(deg + 1);

    // Combine the mesh and basis functions in H1Space2D to define the total global degrees of freedom
    H1Space2D fem(mesh, basis);

    const int ndof = fem.size(); // number of degrees of freedom
    const int N = 2 * ndof;      // total degrees of freedom in [u, v] (U := u + i v)

    auto W = [&]() -> WaveHoltz {
        ivec boundary_faces = mesh.boundary_edges();                 // identify boundary faces
        TraceSpace2D fs(fem, boundary_faces.size(), boundary_faces); // define trace space

        auto a2 = gridfunc(fem, [] __device__(const double X[2]) -> double {
            double aX = alpha(X);
            return aX * aX;
        });

        auto a = trace(fs, [] __device__(const double X[2]) -> double { return alpha(X); });

        auto d_a2 = thrust::raw_pointer_cast(a2.data());
        auto d_a = thrust::raw_pointer_cast(a.data());

        return WaveHoltz(omega, maxvel, d_a2, d_a, fem, fs);
    }();

    thrust::universal_vector<double> U(N, 0.0), B(N, 0.0), GB(N, 0.0);

    double *u = thrust::raw_pointer_cast(U.data());   // device pointer to solution vector
    double *b = thrust::raw_pointer_cast(B.data());   // device pointer to right-hand side vector
    double *Gb = thrust::raw_pointer_cast(GB.data()); // device pointer to G applied to b

    l2_project(b, MassMatrix(fem), [=] __device__(const double X[2]) -> double { return f(X, omega); });

    std::cout << "Solving the Helmholtz equation...\n"
              << "\tomega = " << omega << "\n"
              << "\t#elements = " << mesh.n_elem() << "\n"
              << "\tpolynomial degree = " << deg << "\n"
              << "\t#dof = " << 2 * ndof << std::endl;

    // Solve the system using GMRES
    W.G(b, Gb); // apply G to the right-hand side vector
    SolverResults out = gmres(N, u, &W, Gb, opts);

    double res_norm = [&]() {
        thrust::universal_vector<double> Res(N);
        double *res = thrust::raw_pointer_cast(Res.data());

        ivec boundary_faces = mesh.boundary_edges();
        TraceSpace2D fs(fem, boundary_faces.size(), boundary_faces);

        auto a2 = gridfunc(fem, [] __device__(const double X[2]) -> double {
            double aX = alpha(X);
            return aX * aX;
        });

        auto a = trace(fs, [] __device__(const double X[2]) -> double { return alpha(X); });

        auto d_a2 = thrust::raw_pointer_cast(a2.data());
        auto d_a = thrust::raw_pointer_cast(a.data());

        Helmholtz A(omega, d_a2, d_a, fem, fs);
        A.action(u, res);            // compute residuals
        axpby(N, -1.0, b, 1.0, res); // res = A U - b

        return cuddh::norm(N, res) / cuddh::norm(N, b);
    }();

    std::cout << "Relative residual norm ||A u - b|| / ||b|| = " << res_norm << std::endl;

    // save the solution to a file
    auto xy = fem.physical_coordinates(MemorySpace::HOST);

    auto xyfile = "solution/xy.0000";
    auto solfile = "solution/waveholtz.0000";
    auto resfile = "solution/residuals.0000";

    if (to_file(xyfile, 2 * ndof, xy.data()))
        std::cout << "Saved collocation points to " << xyfile << std::endl;
    else
        std::cerr << "Failed to save collocation points to " << xyfile << std::endl;

    if (to_file(solfile, N, thrust::raw_pointer_cast(U.data())))
        std::cout << "Saved solution to " << solfile << std::endl;
    else
        std::cerr << "Failed to save solution to " << solfile << std::endl;

    if (to_file(resfile, out.res_norm.size(), out.res_norm.data()))
        std::cout << "Saved residuals to " << resfile << std::endl;
    else
        std::cerr << "Failed to save residuals to " << resfile << std::endl;

    return 0;
}