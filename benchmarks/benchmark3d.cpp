#include <algorithm>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "CLI11.hpp"
#include "cuddh.hpp"

using namespace cuddh;

static DDKernelConfig::BlockSize parse_block_size(int block_size);
static bool write_csv_row(const std::string &output_file, const std::string &precision, int degree, int nx, int ny,
                          int nz, double omega, int sx, int sy, int sz, int kernel_block_size, int kernel_tdof,
                          int warmup, int iterations, int n_subdomains, int n_dof, int n_lambda_dof,
                          float helmholtz_avg_ms, float min_ms, float max_ms, float avg_ms, float total_ms);

template <typename scalar_t>
static int run_benchmark(const char *precision, int degree, int nx, int ny, int nz, double omega, int sx, int sy,
                         int sz, DDKernelConfig config, int waveholtz_iterations, int warmup, int iterations,
                         const std::string &output_file)
{
    std::mt19937 gen(42);
    std::uniform_real_distribution<scalar_t> dist(-1, 1);

    Mesh3D mesh = Mesh3D::uniform_cube(nx, -1.0, 1.0, ny, -1.0, 1.0, nz, -1.0, 1.0);
    Basis basis(degree + 1);
    H1Space3D fem(mesh, basis);
    EnsembleSpace3D efem = partition_uniform_cube(fem, {(unsigned int)nx, (unsigned int)ny, (unsigned int)nz},
                                                  {(unsigned int)sx, (unsigned int)sy, (unsigned int)sz});

    GridFunc3D<double> a = gridfunc(fem, [] __device__(double3) -> double { return 1.0; });

    DDSubstructuredOperator3D<scalar_t> F(efem, omega, a, config, waveholtz_iterations);

    auto boundary_faces = mesh.get_boundary_faces();
    TraceSpace3D fs(fem, boundary_faces.size(), boundary_faces);
    Helmholtz3D H(fem, fs, omega, a);

    thrust::universal_vector<scalar_t> lambda_a(F.ndof(), scalar_t(0));
    thrust::universal_vector<scalar_t> lambda_b(F.ndof(), scalar_t(0));

    for (auto &value : lambda_a)
        value = dist(gen);

    scalar_t *in = thrust::raw_pointer_cast(lambda_a.data());
    scalar_t *out = thrust::raw_pointer_cast(lambda_b.data());

    for (int i = 0; i < warmup; ++i)
    {
        F.action(in, out);
        CUDDH_CUDA_CHECK(cudaDeviceSynchronize());
        std::swap(in, out);
    }

    const int n_fem = 2 * fem.size();
    thrust::universal_vector<double> helm_in(n_fem, 0.0);
    thrust::universal_vector<double> helm_out(n_fem, 0.0);
    for (auto &value : helm_in)
        value = static_cast<double>(dist(gen));

    double *helm_x = thrust::raw_pointer_cast(helm_in.data());
    double *helm_y = thrust::raw_pointer_cast(helm_out.data());

    for (int i = 0; i < warmup; ++i)
    {
        H.action(helm_x, helm_y);
        CUDDH_CUDA_CHECK(cudaDeviceSynchronize());
        std::swap(helm_x, helm_y);
    }

    cudaEvent_t start, stop;
    CUDDH_CUDA_CHECK(cudaEventCreate(&start));
    CUDDH_CUDA_CHECK(cudaEventCreate(&stop));

    std::vector<float> runtimes_ms(iterations, 0.0f);
    std::vector<float> helmholtz_runtimes_ms(iterations, 0.0f);
    for (int i = 0; i < iterations; ++i)
    {
        CUDDH_CUDA_CHECK(cudaEventRecord(start));
        F.action(in, out);
        CUDDH_CUDA_CHECK(cudaEventRecord(stop));
        CUDDH_CUDA_CHECK(cudaEventSynchronize(stop));
        CUDDH_CUDA_CHECK(cudaEventElapsedTime(&runtimes_ms[i], start, stop));
        std::swap(in, out);

        CUDDH_CUDA_CHECK(cudaEventRecord(start));
        H.action(helm_x, helm_y);
        CUDDH_CUDA_CHECK(cudaEventRecord(stop));
        CUDDH_CUDA_CHECK(cudaEventSynchronize(stop));
        CUDDH_CUDA_CHECK(cudaEventElapsedTime(&helmholtz_runtimes_ms[i], start, stop));
        std::swap(helm_x, helm_y);
    }

    CUDDH_CUDA_CHECK(cudaEventDestroy(start));
    CUDDH_CUDA_CHECK(cudaEventDestroy(stop));

    const float total_ms = std::accumulate(runtimes_ms.begin(), runtimes_ms.end(), 0.0f);
    const float avg_ms = total_ms / static_cast<float>(iterations);
    const float helmholtz_total_ms = std::accumulate(helmholtz_runtimes_ms.begin(), helmholtz_runtimes_ms.end(), 0.0f);
    const float helmholtz_avg_ms = helmholtz_total_ms / static_cast<float>(iterations);
    const auto [min_it, max_it] = std::minmax_element(runtimes_ms.begin(), runtimes_ms.end());

    auto relative_to_helmholtz = [&](float t_ms) -> float {
        if (helmholtz_avg_ms <= 0.0f)
            return 0.0f;
        return t_ms / helmholtz_avg_ms;
    };

    auto print_time_with_relative = [&](const char *label, float t_ms) {
        std::cout << "  " << std::left << std::setw(18) << label << std::right << std::fixed << std::setprecision(3)
                  << t_ms << " [ms] " << std::setprecision(1) << "(x" << relative_to_helmholtz(t_ms) << ")\n";
    };

    std::cout << "Benchmarking DDSubstructuredOperator3D<" << precision << ">::action\n"
              << "  precision:        " << precision << "\n"
              << "  degree:           " << degree << "\n"
              << "  mesh:             " << nx << " x " << ny << " x " << nz << "\n"
              << "  omega:            " << omega << "\n"
              << "  subdomains dims:  " << sx << " x " << sy << " x " << sz << "\n"
              << "  kernel:           " << F.kernel_str() << "\n"
              << "  #subdomains:      " << efem.size() << "\n"
              << "  #dof:             " << n_fem << "\n"
              << "  #lambda dof:      " << F.ndof() << "\n"
              << "  warmup:           " << warmup << "\n"
              << "  iterations:       " << iterations << "\n"
              << std::fixed << std::setprecision(3) << "  Helmholtz avg time: " << helmholtz_avg_ms << " [ms]\n";

    print_time_with_relative("min time:", *min_it);
    print_time_with_relative("max time:", *max_it);
    print_time_with_relative("avg time:", avg_ms);

    std::cout << std::fixed << std::setprecision(3) << "  total time:        " << total_ms << " [ms]" << std::endl;

    if (!output_file.empty())
    {
        const bool ok = write_csv_row(output_file, precision, degree, nx, ny, nz, omega, sx, sy, sz,
                                      static_cast<int>(config.block_size), config.tdof, warmup, iterations, efem.size(),
                                      n_fem, F.ndof(), helmholtz_avg_ms, *min_it, *max_it, avg_ms, total_ms);
        if (!ok)
            throw std::runtime_error("failed to write output file: " + output_file);
    }

    return 0;
}

int main(int argc, char **argv)
{
    CLI::App app{"Benchmark DDSubstructuredOperator3D::action in 3D"};

    std::string precision = "float";
    int degree = 3;
    std::vector<int> mesh_dims{32, 32, 32};
    std::vector<int> subdomain_dims{4, 4, 2};
    int block_size = 0;
    int tdof = 0;
    int waveholtz_iterations = 2;
    int warmup = 5;
    int iterations = 20;
    std::string output_file;

    app.add_option("-p,--precision", precision, "Scalar type: float or double");
    app.add_option("-d,--degree", degree, "Polynomial degree")->required();
    app.add_option("--mesh", mesh_dims, "Mesh dimensions: nx ny nz")->expected(3)->required();
    app.add_option("--subdomains", subdomain_dims, "Desired subdomain dimensions: sx sy sz")->expected(3)->required();
    app.add_option("--block-size", block_size, "Kernel block size: 0, 256, 512, or 1024");
    app.add_option("--tdof", tdof, "Kernel tdof value");
    app.add_option("--waveholtz-iterations", waveholtz_iterations,
                   "Fixed number of WaveHoltz iterations; -1 uses residual-based stopping");
    app.add_option("--warmup", warmup, "Warmup action calls before timing");
    app.add_option("-n,--iterations", iterations, "Number of timed action calls");
    app.add_option("-o,--output", output_file,
                   "Optional CSV output path. Appends one row and writes a header automatically if needed.");

    CLI11_PARSE(app, argc, argv);

    const int nx = mesh_dims[0];
    const int ny = mesh_dims[1];
    const int nz = mesh_dims[2];
    const int sx = subdomain_dims[0];
    const int sy = subdomain_dims[1];
    const int sz = subdomain_dims[2];

    if (degree < 1)
        throw std::runtime_error("--degree must be positive");
    if (nx < 1 || ny < 1 || nz < 1)
        throw std::runtime_error("--mesh entries must be positive");
    if (sx < 1 || sy < 1 || sz < 1)
        throw std::runtime_error("--subdomains entries must be positive");
    if (precision != "float" && precision != "double")
        throw std::runtime_error("--precision must be either 'float' or 'double'");
    if (tdof < 0 || tdof > 4)
        throw std::runtime_error("--tdof must be between 0 and 4");
    if (waveholtz_iterations != -1 && waveholtz_iterations < 1)
        throw std::runtime_error("--waveholtz-iterations must be positive or -1");
    if (warmup < 0)
        throw std::runtime_error("--warmup must be non-negative");
    if (iterations < 1)
        throw std::runtime_error("--iterations must be positive");

    const double omega = 0.1 * static_cast<double>(std::max(nx, ny)) * static_cast<double>(degree);

    const DDKernelConfig config = {
        .block_size = parse_block_size(block_size),
        .tdof = tdof,
    };

    if (precision == "float")
        return run_benchmark<float>("float", degree, nx, ny, nz, omega, sx, sy, sz, config, waveholtz_iterations,
                                    warmup, iterations, output_file);

    return run_benchmark<double>("double", degree, nx, ny, nz, omega, sx, sy, sz, config, waveholtz_iterations, warmup,
                                 iterations, output_file);
}

static DDKernelConfig::BlockSize parse_block_size(int block_size)
{
    switch (block_size)
    {
        case 0:
            return DDKernelConfig::Default;
        case 256:
            return DDKernelConfig::t256;
        case 512:
            return DDKernelConfig::t512;
        case 1024:
            return DDKernelConfig::t1024;
        default:
            throw std::runtime_error("--block-size must be one of {0, 256, 512, 1024}");
    }
}

static bool write_csv_row(const std::string &output_file, const std::string &precision, int degree, int nx, int ny,
                          int nz, double omega, int sx, int sy, int sz, int kernel_block_size, int kernel_tdof,
                          int warmup, int iterations, int n_subdomains, int n_dof, int n_lambda_dof,
                          float helmholtz_avg_ms, float min_ms, float max_ms, float avg_ms, float total_ms)
{
    std::ofstream out(output_file, std::ios::app);
    if (!out)
        return false;

    std::ifstream in(output_file);
    const bool empty_file = !in.good() || (in.peek() == std::ifstream::traits_type::eof());

    if (empty_file)
    {
        out << "precision,degree,nx,ny,nz,omega,sx,sy,sz,kernel_block_size,kernel_tdof,warmup,iterations,"
            << "n_subdomains,n_dof,n_lambda_dof,helmholtz_avg_ms,min_ms,max_ms,avg_ms,total_ms,"
            << "min_rel_to_helmholtz,max_rel_to_helmholtz,avg_rel_to_helmholtz\n";
    }

    auto rel = [&](float t) -> float {
        if (helmholtz_avg_ms <= 0.0f)
            return 0.0f;
        return t / helmholtz_avg_ms;
    };

    out << precision << ',' << degree << ',' << nx << ',' << ny << ',' << nz << ',' << omega << ',' << sx << ',' << sy
        << ',' << sz << ',' << kernel_block_size << ',' << kernel_tdof << ',' << warmup << ',' << iterations << ','
        << n_subdomains << ',' << n_dof << ',' << n_lambda_dof << ',' << helmholtz_avg_ms << ',' << min_ms << ','
        << max_ms << ',' << avg_ms << ',' << total_ms << ',' << rel(min_ms) << ',' << rel(max_ms) << ',' << rel(avg_ms)
        << '\n';

    return static_cast<bool>(out);
}