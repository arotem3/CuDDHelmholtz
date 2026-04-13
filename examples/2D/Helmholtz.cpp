/**
 * @file Helmholtz.cpp
 * @brief Example driver for solving the Helmholtz equation
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
 * The weak formulation is
 *
 *      a(u, phi) == b(phi)        for all phi in H1(D)
 *
 * The bilinear form a is defined as
 *
 *      a(u, phi) = (grad u, grad phi) - omega^2 (a^2(x) u, phi) - i omega <a(x) u, phi>
 *
 * And the linear operator b is defined b(phi) = (f, phi)
 *
 * The Helmholtz class implements this discretization and solves the problem by solving a linear system with MINRES.
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
 * format. The solution is written to `solution/uv.0000` in binary
 * format.
 *
 * This format can be read and visualized, for example, in Python. See `visualize.py`.
 */

#include "CLI11.hpp"
#include "cuddh.hpp"
#include "examples.hpp"

using namespace cuddh;

/// @brief forcing term, approximate point source
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

/// @brief a(x) = 1/c(x) where c(x) is the wave-speed.
__device__ static double a(const double2 X)
{
    const auto [x, y] = X;
    const double r = max(abs(x), abs(y));

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
    int maxit = 10'000;
    double rtol = 1e-3;
    std::string verbose_str = "progress";

    CLI::App app{"Helmholtz: Direct solver for the 2D Helmholtz equation"};
    app.add_option("-p,--deg", deg, "Polynomial degree of basis functions")->default_val(3);
    app.add_option("-n,--grid", grid, "Grid dimensions: nx [ny] (if ny omitted, ny=nx)")
        ->expected(1, 2)
        ->default_val("32");
    app.add_option("-w,--omega", omega, "Helmholtz frequency (default: 0.1 * nx * deg)");
    app.add_option("--maxit", maxit, "Maximum number of MINRES iterations")->default_val(10'000);
    app.add_option("--rtol", rtol, "Relative tolerance for MINRES")->default_val(1e-3);
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

    const SolverParams opts = {.maxit = maxit, .rtol = rtol, .verbose = verbosity};

    // Assemble the mesh
    Mesh2D mesh = Mesh2D::uniform_rect(nx, -1.0, 1.0, ny, -1.0, 1.0);

    // Construct 1D basis functions. On each element, the 2D basis functions are
    // tensor products of these 1D basis functions.
    Basis basis(deg + 1);

    // The mesh and 1D basis functions are combined in H1Space2D to define the
    // total global degrees of freedom of the problem.
    H1Space2D fem(mesh, basis);
    const int ndof = fem.size(); // # of degrees of freedom

    // The TraceSpace2D is a subspace of the H1Space2D where we apply the boundary conditions
    ivec boundary_faces = mesh.boundary_edges();
    TraceSpace2D fs(fem, boundary_faces.size(), boundary_faces);

    auto coef = gridfunc(fem, [] __device__(double2 x) -> double { return a(x); });

    std::cout << "Solving the Helmholtz equation...\n"
              << "\tomega = " << omega << "\n"
              << "\t#elements = " << mesh.n_elem() << "\n"
              << "\tpolynomial degree = " << deg << "\n"
              << "\t#dof = " << 2 * ndof << "\n";

    Helmholtz A(fem, fs, omega, coef);

    const int N = 2 * ndof; // total degrees of freedom in [u, v] (U := u + i v)

    thrust::universal_vector<double> U(N, 0.0); // solution vector [u; v] initialized to zero
    thrust::universal_vector<double> B(N, 0.0);

    double *u = thrust::raw_pointer_cast(U.data()); // the solution vector [u; v]
    double *b = thrust::raw_pointer_cast(B.data()); // the right hand side b(phi)

    l2_project(b, MassMatrix(fem), [=] __device__(const double2 X) -> double { return f(X, omega); });

    // solve a([u, v], phi) = b(phi)
    std::cout << "\nsolving with MINRES ... \n";
    auto out = minres(u, A, b, opts);
    CUDDH_CUDA_CHECK(cudaDeviceSynchronize());

    // save solution and collocation nodes to file
    auto xy = fem.physical_coordinates(MemorySpace::HOST);

    const char xy_file[] = "solution/xy.0000";
    const char sol_file[] = "solution/uv.0000";
    const char res_file[] = "solution/residuals.0000";

    if (to_file(xy_file, xy.size(), xy.data()))
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
