/**
 * @file DDH.cpp
 * @brief Example driver for solving the Helmholtz equation with the domain decomposition solver
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
 * The DDH class implements this discretization and solves the problem by solving a substructured problem on the
 * skeleton of the domain decomposition. The substructured problem is solved with MINRES and the action of the operator
 * is computed by solving the subdomain problems with the WaveHoltz iterations.
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
 * format. The solution is written to `solution/uv.0000` in binary
 * format.
 *
 * This format can be read and visualized, for example, in Python. See `visualize.py`.
 */

#include <memory>

#include "CLI11.hpp"
#include "cuddh.hpp"
#include "examples.hpp"

using namespace cuddh;

// forcing
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

// variable coefficient
__device__ static double alpha(const double2 X)
{
    const auto [x, y] = X;
    const double r = max(abs(x), abs(y));

    if (r < 0.5)
        return 0.5;
    else
        return 1.0;
}

static int get_n_lambda(const std::unique_ptr<Solver<double>> &ddh);
static std::string get_kernel_str(const std::unique_ptr<Solver<double>> &ddh);

int main(int argc, char *argv[])
{
    int deg = 3;                          // polynomial degree of basis functions
    std::vector<int> grid = {32};         // grid dimensions [nx, ny]
    double omega = -1.0;                  // Helmholtz frequency
    int tdof = 0;                         // one of 0, 1, 2, 3, 4
    int block_size = 0;                   // 0 (default), 256, 512, 1024
    int maxit = 1000;                     // maximum number of GMRES iterations
    double rtol = 1e-3;                   // relative tolerance
    std::string verbose_str = "progress"; // silent | progress | iteration
    std::string subsolver = "waveholtz";  // waveholtz | minres

    CLI::App app{"DDH: Domain decomposition solver for the 2D Helmholtz equation"};
    app.add_option("-p,--deg", deg, "Polynomial degree of basis functions")->default_val(3);
    app.add_option("-n,--grid", grid, "Grid dimensions: nx [ny] (if ny omitted, ny=nx)")
        ->expected(1, 2)
        ->default_val("32");
    app.add_option("-w,--omega", omega, "Helmholtz frequency (default: 0.1 * nx * deg)");
    app.add_option("--tdof", tdof, "DOFs/thread kernel parameter 0 (auto), 1, ..., 4")
        ->default_val(0)
        ->check(CLI::Range(0, 4));
    app.add_option("--block-size", block_size, "Threads per block: 0 (auto), 256, 512, 1024")
        ->default_val(0)
        ->check(CLI::IsMember({0, 256, 512, 1024}));
    app.add_option("--maxit", maxit, "Maximum number of GMRES iterations")->default_val(1000);
    app.add_option("--rtol", rtol, "Relative tolerance for GMRES")->default_val(1e-3);
    app.add_option("-v,--verbose", verbose_str, "Verbosity: silent | progress | iteration")
        ->default_val("progress")
        ->check(CLI::IsMember({"silent", "progress", "iteration"}, CLI::ignore_case));
    app.add_option("--subsolver", subsolver, "Subdomain solver: waveholtz | minres")
        ->default_val("waveholtz")
        ->check(CLI::IsMember({"waveholtz", "minres"}, CLI::ignore_case));
    CLI11_PARSE(app, argc, argv);

    int nx = grid[0];
    int ny = grid.size() > 1 ? grid[1] : grid[0];
    if (omega < 0.0)
        omega = 0.1 * nx * deg;

    DDKernelConfig::BlockSize bs;
    if (block_size == 256)
        bs = DDKernelConfig::t256;
    else if (block_size == 512)
        bs = DDKernelConfig::t512;
    else if (block_size == 1024)
        bs = DDKernelConfig::t1024;
    else
        bs = DDKernelConfig::Default;

    SolverParams::Verbosity verbosity;
    if (CLI::detail::to_lower(verbose_str) == "silent")
        verbosity = SolverParams::Silent;
    else if (CLI::detail::to_lower(verbose_str) == "iteration")
        verbosity = SolverParams::Iteration;
    else
        verbosity = SolverParams::ProgressBar;

    const DDKernelConfig config = {.block_size = bs, .tdof = tdof};

    const SolverParams opts = {.maxit = maxit, .rtol = rtol, .verbose = verbosity};

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
    const int N = 2 * ndof;      // total degrees of freedom in [u, v] (U := u + i v)

    // variable coefficient
    auto a = gridfunc(fem, [] __device__(const double2 X) -> double { return alpha(X); });

    auto ddsolver = [&]() -> std::unique_ptr<Solver<double>> {
        if (subsolver == "minres")
            return std::make_unique<DDH<float, SubdomainSolver::MINRES>>(efem, omega, a, config);
        else
            return std::make_unique<DDH<float>>(efem, omega, a, config);
    }();

    thrust::universal_vector<double> U(N, 0.0);
    thrust::universal_vector<double> B(N, 0.0);

    double *u = thrust::raw_pointer_cast(U.data()); // the solution vector [u; v]
    double *b = thrust::raw_pointer_cast(B.data()); // the right hand side b(phi)

    l2_project(b, MassMatrix(fem), [=] __device__(const double2 X) -> double {
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
              << "\tkernel = {" << get_kernel_str(ddsolver) << "}\n"
              << "\t#lambda = " << get_n_lambda(ddsolver) << std::endl;

    auto out = ddsolver->solve(u, b, opts);

    double res = [&]() -> double {
        ivec boundary_faces = mesh.boundary_edges();                 // identify boundary faces
        TraceSpace2D fs(fem, boundary_faces.size(), boundary_faces); // define trace space

        Helmholtz A(fem, fs, omega, a);

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
    const char sol_file[] = "solution/uv.0000";
    const char res_file[] = "solution/residuals.0000";

    if (to_file(xy_file, xy.size(), xy.data()))
        std::cout << "\ncoordinates written to: " << xy_file << "\n";
    if (to_file(sol_file, N, u))
        std::cout << "Solution written to: " << sol_file << "\n";
    if (to_file(res_file, out.res_norm.size(), out.res_norm.data()))
        std::cout << "Residuals written to: " << res_file << "\n";

    return 0;
}

int get_n_lambda(const std::unique_ptr<Solver<double>> &ddh)
{
    if (auto ddh_ptr = dynamic_cast<DDH<float, SubdomainSolver::MINRES> *>(ddh.get()))
        return ddh_ptr->op().ndof();
    else if (auto ddh_ptr = dynamic_cast<DDH<float> *>(ddh.get()))
        return ddh_ptr->op().ndof();
    return 0;
}

std::string get_kernel_str(const std::unique_ptr<Solver<double>> &ddh)
{
    if (auto ddh_ptr = dynamic_cast<DDH<float, SubdomainSolver::MINRES> *>(ddh.get()))
        return ddh_ptr->op().kernel_str();
    else if (auto ddh_ptr = dynamic_cast<DDH<float> *>(ddh.get()))
        return ddh_ptr->op().kernel_str();
    return std::string("unknown");
}
