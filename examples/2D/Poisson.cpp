/**
 * @file Poisson.cpp
 * @brief Example driver for solving the Poisson equation
 *
 * @details This file is a driver for solving the Poisson equation with
 * Dirichlet boundary conditions:
 *
 *      -div(grad u) == f       in D := [-1, 1]x[-1, 1]
 *          u == g              on boundary of D
 *
 * Define u = v + g where g is an extension of the Dirichlet data to all of D.
 * The weak formulation is
 *
 *      a(v, phi) == b(phi)     for all phi in H1_0(D)
 *
 * The bilinear form a is defined as
 *
 *      a(v, phi) = (grad v, grad phi)
 *
 * And the linear operator b is defined as b(phi) = (f, phi) - (grad g, grad phi),
 *
 * In cuddh, a is computed via StiffnessMatrix::action.
 *
 * To compile & run this program:
 *  (1) From the CuDDHelmholtz directory, compile the library:
 *      cmake .
 *      make cuddh -j
 *  (2) compile the program:
 *      make Poisson
 *  (3) run:
 *      ./examples/Poisson
 *
 * The program will write the collocation points to `solution/xy.0000` in binary
 * format. The solution is written to `solution/u.0000` in binary
 * format.
 *
 * This format can be read and visualized, for example, in Python. See `visualize.py`.
 */

#include "CLI11.hpp"
#include "cuddh.hpp"
#include "examples.hpp"

using namespace cuddh;

__device__ static double f(const double2)
{
    return 1.0;
}

__device__ static double g(const double2 X)
{
    const auto [x, y] = X;
    if (std::abs(x - 1.0) < 1e-12)
        return 1.0 - y * y;
    else if (std::abs(x + 1.0) < 1e-12)
        return y * (1.0 - y * y);
    return 0.0;
}

int main(int argc, char *argv[])
{
    int deg = 3;
    std::vector<int> grid = {15};
    int maxit = 1000;
    double rtol = 1e-3;
    std::string verbose_str = "progress";

    CLI::App app{"Poisson: Solves the Poisson equation with Dirichlet boundary conditions"};
    app.add_option("-p,--deg", deg, "Polynomial degree of basis functions")->default_val(3);
    app.add_option("-n,--grid", grid, "Grid dimensions: nx [ny] (if ny omitted, ny=nx)")
        ->expected(1, 2)
        ->default_val("15");
    app.add_option("--maxit", maxit, "Maximum number of MINRES iterations")->default_val(1000);
    app.add_option("--rtol", rtol, "Relative tolerance for MINRES")->default_val(1e-3);
    app.add_option("-v,--verbose", verbose_str, "Verbosity: silent | progress | iteration")
        ->default_val("progress")
        ->check(CLI::IsMember({"silent", "progress", "iteration"}, CLI::ignore_case));
    CLI11_PARSE(app, argc, argv);

    const int nx = grid[0];
    const int ny = grid.size() > 1 ? grid[1] : grid[0];

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

    std::cout << "Solving the Poisson equation...\n"
              << "\t#elements = " << mesh.n_elem() << "\n"
              << "\tpolynomial degree = " << deg << "\n"
              << "\t#dof = " << ndof << "\n";

    // identify the boundary faces in the mesh in order to define the TraceSpace2D
    // and FaceMassMatrix
    ivec boundary_faces = mesh.boundary_edges();

    // The TraceSpace2D is a subspace of the H1Space2D used to identify the degrees
    // of freedom needed on the boundary of the domain. In particular, we use
    // this object to restrict the solution to H1_0.
    TraceSpace2D fs(fem, boundary_faces.size(), boundary_faces);

    // To manage memory between host and device, we use the HostDeviceArray class.
    host_device_dvec _u(ndof);
    host_device_dvec _b(ndof);

    double *u = _u.device_write(); // the solution vector
    double *b = _b.device_write(); // the right hand side: (f, phi) - (grad g, grad phi)

    // linear system
    StiffnessMatrix A0(fem);
    HomogeneousDirichletOperator2D A(A0, fs);
    DirichletBC2D bc(fs);

    // set up right hand side
    l2_project(b, MassMatrix(fem), [] __device__(const double2 x) -> double { return f(x); }); // (f, phi)

    bc.set([] __device__(const double2 x) -> double { return g(x); });
    bc.apply_rhs(A0, b); // b <- orth(b) - A0 * E(q)

    // solve for u (without boundary conditions)
    std::cout << "\nsolving with minres... \n";
    auto out = minres(u, A, b, opts);

    // add extension to u (now u satisfies Dirichlet BCs)
    bc.recover_solution(u);

    // copy to host
    const double *h_u = _u.host_read();

    // save solution and collocation nodes to file
    auto xy = fem.physical_coordinates(MemorySpace::HOST);

    const char xy_file[] = "solution/xy.0000";
    const char sol_file[] = "solution/u.0000";
    const char res_file[] = "solution/residuals.0000";

    if (to_file(xy_file, xy.size(), xy.data()))
        std::cout << "Coordinates written to: " << xy_file << "\n";
    else
        std::cerr << "Failed to write coordinates to: " << xy_file << "\n";

    if (to_file(sol_file, ndof, h_u))
        std::cout << "Solution written to: " << sol_file << "\n";
    else
        std::cerr << "Failed to write solution to: " << sol_file << "\n";

    if (to_file(res_file, out.res_norm.size(), out.res_norm.data()))
        std::cout << "Residuals written to: " << res_file << "\n";
    else
        std::cerr << "Failed to write residuals to: " << res_file << "\n";

    return 0;
}