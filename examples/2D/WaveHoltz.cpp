/**
 * @file WaveHoltz.cpp
 * @brief Example driver for solving the Helmholtz equation with the WaveHoltz solver.
 *
 * @details This file is a driver for solving the Helmholtz equation with
 * approximate absorbing boundary conditions:
 *
 *      -div(grad u) - omega^2 a^2(x) u == f    in  D := [-1, 1]x[-1, 1]
 *      i a(x) omega u + du/dn == 0             on boundary of D
 *
 * Here omega is the frequency. We assume f is real valued, and u is complex
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
 * format. The solution is written to `solution/uv.0000` in binary
 * format.
 *
 * This format can be read and visualized, for example, in Python. See `visualize.py`.
 */

#include <format>

#include "CLI11.hpp"
#include "cuddh.hpp"
#include "examples.hpp"

using namespace cuddh;

__device__ static double f(const double2 X, double omega)
{
    const auto [x, y] = X;
    double s = omega * omega;

    double r = (x + 0.5) * (x + 0.5) + y * y;
    double F = s / M_PI * std::exp(-s * r);

    r = (x - 0.5) * (x - 0.5) + (y + 0.5) * (y + 0.5);
    F += s / M_PI * std::exp(-s * r);
    return F;
}

__device__ static double alpha(const double2 X)
{
    const auto [x, y] = X;
    const double r = std::max(std::abs(x), std::abs(y));

    if (r < 0.5)
        return 0.5;
    else
        return 1.0;
}

int main(int argc, char *argv[])
{
    int deg = 3;
    std::vector<int> grid = {32};
    double omega = -1.0;
    int kdim = 50;
    int edim = 20;
    int maxit = 1000;
    double rtol = 1e-3;
    std::string verbose_str = "progress";

    CLI::App app{"WaveHoltz: WaveHoltz solver for the 2D Helmholtz equation"};
    app.add_option("-p,--deg", deg, "Polynomial degree of basis functions")->default_val(3);
    app.add_option("-n,--grid", grid, "Grid dimensions: nx [ny] (if ny omitted, ny=nx)")
        ->expected(1, 2)
        ->default_val("32");
    app.add_option("-w,--omega", omega, "Helmholtz frequency (default: 0.1 * nx * deg)")->default_val(-1.0);
    app.add_option("-k,--kdim", kdim, "Krylov dimension for GCRO")->default_val(50);
    app.add_option("-e,--edim", edim, "Deflation space dimension for GCRO")->default_val(20);
    app.add_option("--maxit", maxit, "Maximum number of iterations")->default_val(1000);
    app.add_option("--rtol", rtol, "Relative tolerance")->default_val(1e-3);
    app.add_option("-v,--verbose", verbose_str, "Verbosity: silent | progress | iteration")
        ->default_val("progress")
        ->check(CLI::IsMember({"silent", "progress", "iteration"}, CLI::ignore_case));
    CLI11_PARSE(app, argc, argv);

    const int nx = grid[0];
    const int ny = grid.size() > 1 ? grid[1] : grid[0];
    if (omega < 0.0)
        omega = 0.1 * nx * deg;

    SolverParams::Verbosity verbosity;
    if (CLI::detail::to_lower(verbose_str) == "silent")
        verbosity = SolverParams::Silent;
    else if (CLI::detail::to_lower(verbose_str) == "iteration")
        verbosity = SolverParams::Iteration;
    else
        verbosity = SolverParams::ProgressBar;

    const SolverParams opts = {
        .maxit = maxit,
        .rtol = rtol,
        .verbose = verbosity,
    };

    // Create a uniform rectangular mesh
    Mesh2D mesh = Mesh2D::uniform_rect(nx, -1.0, 1.0, ny, -1.0, 1.0);

    // Construct 1D basis functions
    Basis basis(deg + 1);

    // Combine the mesh and basis functions in H1Space2D to define the total global degrees of freedom
    H1Space2D fem(mesh, basis);

    ivec boundary_faces = mesh.boundary_edges();                 // identify boundary faces
    TraceSpace2D fs(fem, boundary_faces.size(), boundary_faces); // define trace space

    const int ndof = fem.size(); // number of degrees of freedom
    const int N = 2 * ndof;      // total degrees of freedom in [u, v] (U := u + i v)

    auto a = gridfunc(fem, [] __device__(double2 x) -> double { return alpha(x); });

    WaveHoltz W(fem, fs, omega, a);

    thrust::universal_vector<double> U(N, 0.0), B(N, 0.0), GB(N, 0.0);

    double *u = thrust::raw_pointer_cast(U.data());   // device pointer to solution vector
    double *b = thrust::raw_pointer_cast(B.data());   // device pointer to right-hand side vector
    double *Gb = thrust::raw_pointer_cast(GB.data()); // device pointer to G applied to b

    l2_project(b, MassMatrix(fem), [=] __device__(const double2 X) -> double { return f(X, omega); });

    std::cout << "Solving the Helmholtz equation...\n"
              << "\tomega = " << omega << "\n"
              << "\t#elements = " << mesh.n_elem() << "\n"
              << "\tpolynomial degree = " << deg << "\n"
              << "\t#dof = " << 2 * ndof << std::endl;

    // Solve the system using GMRES
    W.G(b, Gb); // apply G to the right-hand side vector
    SolverResults out = GCRO<double>(W, nullptr, kdim, edim).solve(u, Gb, opts);

    double res_norm = [&]() {
        thrust::universal_vector<double> Res(N);
        double *res = thrust::raw_pointer_cast(Res.data());

        Helmholtz A(fem, fs, omega, a);
        A.action(u, res);                 // compute residuals
        dla::axpby(N, -1.0, b, 1.0, res); // res = A U - b

        return dla::norm(N, res) / dla::norm(N, b);
    }();

    CUDDH_CUDA_CHECK(cudaDeviceSynchronize());

    std::cout << std::format("Helmholtz residual |b - A u| / |b| ~ {:.2e}", res_norm) << std::endl;

    // save the solution to a file
    auto xy = fem.physical_coordinates(MemorySpace::HOST);

    auto xyfile = "solution/xy.0000";
    auto solfile = "solution/uv.0000";
    auto resfile = "solution/residuals.0000";

    if (to_file(xyfile, xy.size(), xy.data()))
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
