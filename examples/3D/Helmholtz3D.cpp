/**
 * @file Helmholtz3D.cpp
 * @brief Example driver for solving the 3D Helmholtz equation
 *
 * @details This file is a driver for solving the Helmholtz equation with
 * approximate absorbing boundary conditions:
 *
 *      -div(grad u) - omega^2 a^2(x) u == f    in  D := [-1, 1]^3
 *      -i omega a(x) u + du/dn == 0            on boundary of D
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
 *      a(u, phi) = (grad u, grad phi) - omega^2 (a^2(x) u, phi) - i omega <a(x) v, phi>.
 *
 * And the linear operator b is defined b(phi) = (f, phi).
 *
 * The Helmholtz3D class implements the operator a(*, phi).
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

/// @brief alpha(x) = 1/c(x) where c(x) is the wave-speed.
__device__ static double alpha(double3 x)
{
    const double r = max(abs(x.x), abs(x.y));
    return (r < 0.5) ? 0.5 : 1.0;
}

int main(int argc, char *argv[])
{
    int deg = 3;
    std::vector<int> grid = {16};
    double omega = -1.0;
    int maxit = 10'000;
    double rtol = 1e-3;
    std::string verbose_str = "progress";

    CLI::App app{"Helmholtz3D: Direct solver for the 3D Helmholtz equation"};
    app.add_option("-p,--deg", deg, "Polynomial degree of basis functions")->default_val(3);
    app.add_option("-n,--grid", grid, "Grid dimensions: nx [ny [nz]] (if omitted, ny=nz=nx)")
        ->expected(1, 3)
        ->default_val("16");
    app.add_option("-w,--omega", omega, "Helmholtz frequency (default: 0.1 * nx * deg)");
    app.add_option("--maxit", maxit, "Maximum number of MINRES iterations")->default_val(10'000);
    app.add_option("--rtol", rtol, "Relative tolerance for MINRES")->default_val(1e-3);
    app.add_option("-v,--verbose", verbose_str, "Verbosity: silent | progress | iteration")
        ->default_val("progress")
        ->check(CLI::IsMember({"silent", "progress", "iteration"}, CLI::ignore_case));
    CLI11_PARSE(app, argc, argv);

    const int nx = grid[0];
    const int ny = grid.size() > 1 ? grid[1] : grid[0];
    const int nz = grid.size() > 2 ? grid[2] : grid[0];
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
    Mesh3D mesh = Mesh3D::uniform_cube(nx, -1.0, 1.0, ny, -1.0, 1.0, nz, -1.0, 1.0);

    // Construct 1D basis functions
    Basis basis(deg + 1);

    // Combine the mesh and basis functions in H1Space3D to define the global DOFs
    H1Space3D fem(mesh, basis);
    const int ndof = fem.size();
    const int N = 2 * ndof;

    std::cout << "Solving the 3D Helmholtz equation...\n"
              << "\tomega = " << omega << "\n"
              << "\t#elements = " << mesh.n_elem() << "\n"
              << "\tpolynomial degree = " << deg << "\n"
              << "\t#dof = " << N << "\n";

    // Identify boundary faces and construct the trace space
    auto boundary_faces = mesh.get_boundary_faces();
    TraceSpace3D fs(fem, boundary_faces.size(), boundary_faces);

    auto a = gridfunc(fem, [=] __device__(double3 x) -> double { return alpha(x); });

    Helmholtz3D A(fem, fs, omega, a);

    thrust::universal_vector<double> U(N, 0.0); // solution vector [u; v] initialized to zero
    thrust::universal_vector<double> B(N, 0.0); // right-hand side [b; 0]

    double *u = thrust::raw_pointer_cast(U.data());
    double *b = thrust::raw_pointer_cast(B.data());

    l2_project(MassMatrix3D(fem), [=] __device__(double3 x) -> double { return f(x, omega); }, b);

    // solve a([u, v], phi) = b(phi)
    std::cout << "\nsolving with MINRES ... \n";
    auto out = minres(u, A, b, opts);
    CUDDH_CUDA_CHECK(cudaDeviceSynchronize());

    // save solution and collocation nodes to file
    auto coo = fem.physical_coordinates(MemorySpace::HOST);

    const char coo_file[] = "solution/coo.0000";
    const char sol_file[] = "solution/uv.0000";
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
