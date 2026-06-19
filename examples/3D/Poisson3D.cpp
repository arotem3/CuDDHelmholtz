/**
 * @file Poisson3D.cpp
 * @brief Example driver for solving the Poisson equation in 3D
 *
 * @details This file is a driver for solving the Poisson equation with
 * Dirichlet boundary conditions:
 *
 *      -div(grad u) == f       in D := [-1, 1]^3
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
 *      make Poisson3D
 *  (3) run:
 *      ./examples/Poisson3D
 *
 * The program will write the collocation points to `solution/coo.0000` in binary
 * format. The solution is written to `solution/u.0000` in binary
 * format.
 *
 * This format can be read and visualized, for example, in Python. See `visualize.py`.
 */

#include "CLI11.hpp"
#include "cuddh.hpp"
#include "examples.hpp"

using namespace cuddh;

__device__ static double f(double3 r)
{
    return -4.0;
}

__device__ static double g(double3 r)
{
    return 3.0 * r.x * r.x - r.y * r.y + r.z;
}

int main(int argc, char *argv[])
{
    int deg = 3;
    std::vector<int> grid = {10};
    int maxit = 500;
    double rtol = 1e-3;
    std::string verbose_str = "progress";

    CLI::App app{"Poisson3D: Solves the 3D Poisson equation with Dirichlet boundary conditions"};
    app.add_option("-p,--deg", deg, "Polynomial degree of basis functions")->default_val(3);
    app.add_option("-n,--grid", grid, "Grid dimensions: nx [ny [nz]] (if omitted, ny=nz=nx)")
        ->expected(1, 3)
        ->default_val("10");
    app.add_option("--maxit", maxit, "Maximum number of MINRES iterations")->default_val(500);
    app.add_option("--rtol", rtol, "Relative tolerance for MINRES")->default_val(1e-3);
    app.add_option("-v,--verbose", verbose_str, "Verbosity: silent | progress | iteration")
        ->default_val("progress")
        ->check(CLI::IsMember({"silent", "progress", "iteration"}, CLI::ignore_case));
    CLI11_PARSE(app, argc, argv);

    const int nx = grid[0];
    const int ny = grid.size() > 1 ? grid[1] : grid[0];
    const int nz = grid.size() > 2 ? grid[2] : grid[0];

    SolverParams::Verbosity verbosity;
    if (CLI::detail::to_lower(verbose_str) == "silent")
        verbosity = SolverParams::Silent;
    else if (CLI::detail::to_lower(verbose_str) == "iteration")
        verbosity = SolverParams::Iteration;
    else
        verbosity = SolverParams::ProgressBar;

    SolverParams opts = {.maxit = maxit, .rtol = rtol, .verbose = verbosity};

    // create the mesh
    Mesh3D mesh = Mesh3D::uniform_cube(nx, -1.0, 1.0, ny, -1.0, 1.0, nz, -1.0, 1.0);

    // Construct 1D basis functions. On each element, the 3D basis functions are
    // tensor products of these 1D basis functions.
    Basis basis(deg + 1);

    // The mesh and 1D basis functions are combined in H1Space3D to define the
    // total global degrees of freedom of the problem.
    H1Space3D fem(mesh, basis);
    const int ndof = fem.size(); // # of degrees of freedom

    std::cout << "Solving the Poisson equation...\n"
              << "\t#elements = " << mesh.n_elem() << "\n"
              << "\tpolynomial degree = " << deg << "\n"
              << "\t#dof = " << ndof << "\n";

    // identify the boundary faces in the mesh in order to define the TraceSpace3D
    // and FaceMassMatrix
    auto boundary_faces = mesh.get_boundary_faces();

    // The TraceSpace3D is a subspace of the H1Space3D used to identify the degrees
    // of freedom needed on the boundary of the domain. In particular, we use
    // this object to restrict the solution to H1_0.
    TraceSpace3D tr(fem, boundary_faces.size(), boundary_faces);
    // To manage memory between host and device, we use the HostDeviceArray class.
    host_device_dvec _u(ndof); // the solution vector
    host_device_dvec _b(ndof); // the right hand side: (f, phi) - (grad g, grad phi)

    double *u = _u.device_write();
    double *b = _b.device_write();

    // linear system
    StiffnessMatrix3D A0(fem);
    HomogeneousDirichletOperator3D A(A0, tr);
    DirichletBC3D bc(tr);

    // set the right hand side
    l2_project(MassMatrix3D(fem), [] __device__(double3 r) { return f(r); }, b); // (f, phi)

    bc.set([] __device__(double3 r) { return g(r); });
    bc.apply_rhs(A0, b); // b <- orth(b) - A0 * E(q)

    // solve the linear system
    std::cout << "\nsolving with minres...\n";
    auto out = minres(u, A, b, opts);

    // add extension to recover the full solution
    bc.recover_solution(u);

    // compare against exact
    host_device_dvec _ue(ndof);
    double *ue = _ue.device_write();
    auto x = fem.physical_coordinates(MemorySpace::DEVICE);
    forall(ndof, [=] __device__(int i) { ue[i] = g(x[i]); });
    std::cout << "\nL2 error ~ " << l2_dist(M, u, ue) << "\n\n";

    // copy to host
    const double *h_u = _u.host_read();
    x = fem.physical_coordinates(MemorySpace::HOST);

    // save solution
    const char coo_file[] = "solution/coo.0000";
    const char u_file[] = "solution/u.0000";
    const char res_file[] = "solution/residuals.0000";

    if (to_file(coo_file, ndof, x.data()))
        std::cout << "Saved collocation points to " << coo_file << std::endl;
    else
        std::cerr << "Failed to save collocation points to " << coo_file << std::endl;

    if (to_file(u_file, ndof, h_u))
        std::cout << "Saved solution to " << u_file << std::endl;
    else
        std::cerr << "Failed to save solution to " << u_file << std::endl;

    if (to_file(res_file, out.res_norm.size(), out.res_norm.data()))
        std::cout << "Saved residuals to " << res_file << std::endl;
    else
        std::cerr << "Failed to save residuals to " << res_file << std::endl;
    return 0;
}