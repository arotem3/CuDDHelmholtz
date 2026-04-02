#include <format>

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

int main()
{
    const int deg = 1;
    const int nx = 32;
    const double omega = 2 * M_PI * nx / 10;

    const DDKernelConfig config = {
        .block_size = DDKernelConfig::Default, // one of Default, t256, t512, t1024
        .tdof = 1                              // one of 0, 1, 2, 3, 4
    };

    const SolverParams opts = {
        .maxit = 1000,                       // maximum number of iterations for DDH solver
        .rtol = 1e-5,                        // relative tolerance. GMRES stops when ||b-A*x|| < tol*||b||
        .verbose = SolverParams::ProgressBar // Silent, ProgressBar, Iteration
    };

    Mesh3D mesh = Mesh3D::uniform_cube(nx, -1.0, 1.0, nx, -1.0, 1.0, nx, -1.0, 1.0);

    Basis basis(deg + 1);

    H1Space3D fem(mesh, basis);
    EnsembleSpace3D efem = partition_uniform_cube(fem, {nx, nx, nx}, {4, 2, 2});

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
              << "\t#lambda = " << ddh.n_lambda() << std::endl;

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
    auto solfile = "solution/waveholtz.0000";
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