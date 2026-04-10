/**
 * @file DDH3D.cpp
 * @brief Example driver for solving the 3D Helmholtz equation with the domain decomposition Helmholtz (DDH) solver.
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
 * The DDH3D class implements solves the Helmholtz equation using a domain decomposition approach.
 *
 * To compile & run this program:
 *  (1) From the CuDDHelmholtz directory, compile the library:
 *      cmake .
 *      make cuddh -j
 *  (2) compile the program:
 *      make DDH3D
 *  (3) run:
 *      ./examples/DDH3D
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
    const double r = x.x * x.x + x.y * x.y + x.z * x.z;
    return (r < 0.0625) ? 0.2 : 1.0;
}

int main(int argc, char *argv[])
{
    int deg = 3;                          // polynomial degree of basis functions
    std::vector<int> grid = {16};         // grid dimensions [nx, ny, nz]
    double omega = -1.0;                  // Helmholtz frequency
    int tdof = 0;                         // one of 0, 1, 2, 3, 4
    int block_size = 0;                   // 0 (default), 256, 512, 1024
    int maxit = 500;                      // maximum number of GMRES iterations
    double rtol = 1e-3;                   // relative tolerance
    std::string verbose_str = "progress"; // silent | progress | iteration

    CLI::App app{"DDH3D: Domain decomposition solver for the 3D Helmholtz equation"};
    app.add_option("-p,--deg", deg, "Polynomial degree of basis functions")->default_val(3);
    app.add_option("-n,--grid", grid, "Grid dimensions: nx [ny [nz]] (if omitted, ny=nz=nx)")
        ->expected(1, 3)
        ->default_val("16");
    app.add_option("-w,--omega", omega, "Helmholtz frequency (default: 0.1 * nx * deg)");
    app.add_option("--tdof", tdof, "DOFs/thread kernel parameter 0 (auto), 1, ..., 4")
        ->default_val(0)
        ->check(CLI::Range(0, 4));
    app.add_option("--block-size", block_size, "Threads per block: 0 (auto), 256, 512, 1024")
        ->default_val(0)
        ->check(CLI::IsMember({0, 256, 512, 1024}));
    app.add_option("--maxit", maxit, "Maximum number of GMRES iterations")->default_val(500);
    app.add_option("--rtol", rtol, "Relative tolerance for GMRES")->default_val(1e-3);
    app.add_option("-v,--verbose", verbose_str, "Verbosity: silent | progress | iteration")
        ->default_val("progress")
        ->check(CLI::IsMember({"silent", "progress", "iteration"}, CLI::ignore_case));
    CLI11_PARSE(app, argc, argv);

    int nx = grid[0];
    int ny = grid.size() > 1 ? grid[1] : grid[0];
    int nz = grid.size() > 2 ? grid[2] : grid[0];
    if (omega <= 0.0)
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

    Mesh3D mesh = Mesh3D::uniform_cube(nx, -1.0, 1.0, ny, -1.0, 1.0, nz, -1.0, 1.0);

    Basis basis(deg + 1);

    H1Space3D fem(mesh, basis);
    EnsembleSpace3D efem = partition_uniform_cube(fem, {(unsigned)nx, (unsigned)ny, (unsigned)nz});

    const int ndof = fem.size();
    const int N = 2 * ndof;

    thrust::universal_vector<double> U(N);
    thrust::universal_vector<double> b(N);
    thrust::universal_vector<double> a(ndof);

    double *u_U = U.data().get();
    double *u_b = b.data().get();
    double *u_a = a.data().get();

    MassMatrix3D M(fem);
    l2_project(M, [=] __device__(double3 x) -> double { return f(x, omega); }, u_b);

    gridfunc(fem, [=] __device__(double3 x) -> double { return alpha(x); }, u_a);

    DDH3D<float> ddh(omega, u_a, fem, efem, config);

    std::cout << "Solving the Helmholtz equation...\n"
              << "\tomega = " << omega << "\n"
              << "\t#elements = " << mesh.n_elem() << "\n"
              << "\tpolynomial degree = " << deg << "\n"
              << "\t#dof = " << 2 * ndof << "\n"
              << "\t#subdomains = " << efem.size() << "\n"
              << "\tmax #elements / subdomain = " << efem.max_n_elem() << "\n"
              << "\tmax #dof / subdomain = " << efem.max_size() << "\n"
              << "\t#lambda = " << ddh.op().ndof() << "\n"
              << "\tkernel = {" << ddh.op().kernel_str() << "}" << std::endl;

    auto out = ddh.solve(u_U, u_b, opts);

    const double residual = [&]() {
        auto boundary_faces = mesh.get_boundary_faces();
        TraceSpace3D fs(fem, boundary_faces.size(), boundary_faces);

        host_device_dvec a2x(ndof);
        host_device_dvec ax(fs.size());

        gridfunc(
            fem,
            [=] __device__(double3 x) -> double {
                double aX = alpha(x);
                return aX * aX;
            },
            a2x.device_write());

        trace(fs, [=] __device__(double3 x) -> double { return alpha(x); }, ax.device_write());

        Helmholtz3D A(omega, a2x.device_read(), ax.device_read(), fem, fs);

        host_device_dvec Au(N);
        double *d_Au = Au.device_write();

        A.action(u_U, d_Au);
        return dla::dist(N, d_Au, u_b) / dla::norm(N, u_b);
    }();

    std::cout << "Helmholtz residual |b - Au| / |b| ~ " << residual << "\n";

    CUDDH_CUDA_CHECK(cudaDeviceSynchronize());

    auto coo = fem.physical_coordinates(MemorySpace::HOST);

    auto coofile = "solution/coo.0000";
    auto solfile = "solution/uv.0000";
    auto resfile = "solution/residuals.0000";

    if (to_file(coofile, coo.size(), coo.data()))
        std::cout << "Saved collocation points to " << coofile << std::endl;
    else
        std::cerr << "Failed to save collocation points to " << coofile << std::endl;

    if (to_file(solfile, U.size(), u_U))
        std::cout << "Saved solution to " << solfile << std::endl;
    else
        std::cerr << "Failed to save solution to " << solfile << std::endl;

    if (to_file(resfile, out.res_norm.size(), out.res_norm.data()))
        std::cout << "Saved residuals to " << resfile << std::endl;
    else
        std::cerr << "Failed to save residuals to " << resfile << std::endl;

    return 0;
}