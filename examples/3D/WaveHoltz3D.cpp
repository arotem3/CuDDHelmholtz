/**
 * @file WaveHoltz3D.cpp
 * @brief Example driver for solving the 3D Helmholtz equation with the WaveHoltz solver.
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
 * The WaveHoltz3D class implements the finite element discretization of this equation,
 * but is actually solving the wave equation at each iteration.
 *
 * To compile & run this program:
 *  (1) From the CuDDHelmholtz directory, compile the library:
 *      cmake .
 *      make cuddh -j
 *  (2) compile the program:
 *      make WaveHoltz3D
 *  (3) run:
 *      ./examples/WaveHoltz3D
 *
 * The program will write the collocation points to `solution/coo.0000` in binary
 * format. The solution is written to `solution/uv.0000` in binary
 * format. The residuals from the GMRES iterations are written to `solution/residuals.0000`
 * in binary format.
 *
 * This format can be read and visualized, for example, in Python. See `visualize.py`.
 */

#include <format>

#include "CLI11.hpp"
#include "cuddh.hpp"
#include "examples.hpp"

using namespace cuddh;

__device__ static double f(double3 x, double omega)
{
    double s = omega * omega;
    double r1 = (x.x - 0.5) * (x.x - 0.5) + x.y * x.y + x.z * x.z;
    double F1 = std::pow(s / M_PI, 1.5) * std::exp(-s * r1);

    double r2 = (x.x + 0.7) * (x.x + 0.7) + (x.y + 0.7) * (x.y + 0.7) + x.z * x.z;
    double F2 = std::pow(s / M_PI, 1.5) * std::exp(-s * r2);

    return F1 + F2;
}

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
    int kdim = 20;
    int maxit = 500;
    double rtol = 1e-3;
    std::string verbose_str = "progress";

    CLI::App app{"WaveHoltz3D: WaveHoltz solver for the 3D Helmholtz equation"};
    app.add_option("-p,--deg", deg, "Polynomial degree of basis functions")->default_val(3);
    app.add_option("-n,--grid", grid, "Grid dimensions: nx [ny [nz]] (if omitted, ny=nz=nx)")
        ->expected(1, 3)
        ->default_val("16");
    app.add_option("-w,--omega", omega, "Helmholtz frequency (default: 0.1 * nx * deg)");
    app.add_option("-k,--kdim", kdim, "Krylov dimension for GMRES")->default_val(20);
    app.add_option("--maxit", maxit, "Maximum number of GMRES iterations")->default_val(500);
    app.add_option("--rtol", rtol, "Relative tolerance for GMRES")->default_val(1e-3);
    app.add_option("-v,--verbose", verbose_str, "Verbosity: silent | progress | iteration")
        ->default_val("progress")
        ->check(CLI::IsMember({"silent", "progress", "iteration"}, CLI::ignore_case));
    CLI11_PARSE(app, argc, argv);

    const int nx = grid[0];
    const int ny = grid.size() > 1 ? grid[1] : grid[0];
    const int nz = grid.size() > 2 ? grid[2] : grid[0];
    if (omega <= 0.0)
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
    Mesh3D mesh = Mesh3D::uniform_cube(nx, -1.0, 1.0, ny, -1.0, 1.0, nz, -1.0, 1.0);

    // Construct 1D basis functions
    Basis basis(deg + 1);

    // Combine the mesh and basis functions in H1Space3D to define the total global degrees of freedom
    H1Space3D fem(mesh, basis);

    const int ndof = fem.size(); // number of degrees of freedom
    const int N = 2 * ndof;      // total degrees of freedom in [u, v] (U := u + i v)

    auto boundary_faces = mesh.get_boundary_faces();             // identify boundary faces
    TraceSpace3D fs(fem, boundary_faces.size(), boundary_faces); // define trace space

    host_device_dvec U(N);  // solution vector
    host_device_dvec b(N);  // forcing term
    host_device_dvec Gb(N); // WaveHoltz G applied to b

    double *d_U = U.device_write();   // device pointer to solution vector
    double *d_b = b.device_write();   // device pointer to right-hand side vector
    double *d_Gb = Gb.device_write(); // device pointer to G applied to b

    // compute b = (f, phi)
    l2_project(MassMatrix3D(fem), [=] __device__(double3 x) -> double { return f(x, omega); }, d_b);

    // build variable coefficient GridFunc3D
    auto a = gridfunc(fem, [=] __device__(double3 x) -> double { return alpha(x); });

    // Initialize the WaveHoltz operator
    WaveHoltz3D W(fem, fs, omega, a);

    W.G(d_b, d_Gb); // apply G to the right-hand side vector

    std::cout << "Solving the Helmholtz equation...\n"
              << "\tomega = " << omega << "\n"
              << "\t#elements = " << mesh.n_elem() << "\n"
              << "\tpolynomial degree = " << deg << "\n"
              << "\t#dof = " << N << std::endl;

    // Solve the system using GMRES
    SolverResults out = gmres(d_U, W, d_Gb, kdim, nullptr, opts);

    double res = [&]() {
        Helmholtz3D A(fem, fs, omega, a);

        thrust::device_vector<double> Au(N);
        double *d_Au = thrust::raw_pointer_cast(Au.data());
        A.action(d_U, d_Au);

        return dla::dist(N, d_Au, d_b) / dla::norm(N, d_b);
    }();

    std::cout << std::format("Helmholtz residual |b - Ax| / |b| ~ {:.2e}", res) << std::endl;

    // save the solution to a file
    auto coo = fem.physical_coordinates(MemorySpace::HOST);

    auto coofile = "solution/coo.0000";
    auto solfile = "solution/uv.0000";
    auto resfile = "solution/residuals.0000";

    if (to_file(coofile, coo.size(), coo.data()))
        std::cout << "Saved collocation points to " << coofile << std::endl;
    else
        std::cerr << "Failed to save collocation points to " << coofile << std::endl;

    if (to_file(solfile, U.size(), U.host_read()))
        std::cout << "Saved solution to " << solfile << std::endl;
    else
        std::cerr << "Failed to save solution to " << solfile << std::endl;

    if (to_file(resfile, out.res_norm.size(), out.res_norm.data()))
        std::cout << "Saved residuals to " << resfile << std::endl;
    else
        std::cerr << "Failed to save residuals to " << resfile << std::endl;

    return 0;
}