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
static float percentile_ms(std::vector<float> samples, double q);
static bool write_csv_row(const std::string &output_file, const std::string &precision, int degree, int nx, int ny,
                          double omega, int sx, int sy, int kernel_block_size, int kernel_tdof, int warmup,
                          int iterations, int n_subdomains, int n_dof, int n_lambda_dof, float waveholtz_min_ms,
                          float waveholtz_max_ms, float waveholtz_avg_ms, float waveholtz_total_ms,
                          float waveholtz_p10_ms, float waveholtz_p50_ms, float waveholtz_p90_ms, float minres_min_ms,
                          float minres_max_ms, float minres_avg_ms, float minres_total_ms, float minres_p10_ms,
                          float minres_p50_ms, float minres_p90_ms);

template <typename scalar_t>
static int run_benchmark(const char *precision, int degree, int nx, int ny, int sx, int sy, DDKernelConfig config,
                         int warmup, int iterations, const std::string &output_file)
{
    std::mt19937 gen(42);
    std::uniform_real_distribution<scalar_t> dist(-1, 1);

    const double omega = 0.1 * static_cast<double>(std::max(nx, ny)) * static_cast<double>(degree);

    Mesh2D mesh = Mesh2D::uniform_rect(nx, -1.0, 1.0, ny, -1.0, 1.0);
    Basis basis(degree + 1);
    H1Space2D fem(mesh, basis);
    EnsembleSpace efem = partition_uniform_rect(fem, {nx, ny}, {sx, sy});

    auto a = gridfunc(fem, [] __device__(const double X[2]) -> double { return 1.0; });
    double *d_a = thrust::raw_pointer_cast(a.data());

    DDSubstructuredOperator<scalar_t, SubdomainSolver::WaveHoltz> F_wh(omega, d_a, fem, efem, config);
    DDSubstructuredOperator<scalar_t, SubdomainSolver::MINRES> F_mr(omega, d_a, fem, efem, config);

    if (F_wh.size() != F_mr.size())
        throw std::runtime_error("internal error: solver variants produced different lambda dimensions");

    thrust::universal_vector<scalar_t> lambda_a(F_wh.size(), scalar_t(0));
    thrust::universal_vector<scalar_t> lambda_wh_out(F_wh.size(), scalar_t(0));
    thrust::universal_vector<scalar_t> lambda_mr_out(F_wh.size(), scalar_t(0));

    for (auto &value : lambda_a)
        value = dist(gen);

    scalar_t *in = thrust::raw_pointer_cast(lambda_a.data());
    scalar_t *wh_out = thrust::raw_pointer_cast(lambda_wh_out.data());
    scalar_t *mr_out = thrust::raw_pointer_cast(lambda_mr_out.data());

    for (int i = 0; i < warmup; ++i)
    {
        F_wh.action(in, wh_out);
        F_mr.action(in, mr_out);
        CUDDH_CUDA_CHECK(cudaDeviceSynchronize());
    }

    cudaEvent_t start, stop;
    CUDDH_CUDA_CHECK(cudaEventCreate(&start));
    CUDDH_CUDA_CHECK(cudaEventCreate(&stop));

    std::vector<float> waveholtz_runtimes_ms(iterations, 0.0f);
    std::vector<float> minres_runtimes_ms(iterations, 0.0f);
    for (int i = 0; i < iterations; ++i)
    {
        CUDDH_CUDA_CHECK(cudaEventRecord(start));
        F_wh.action(in, wh_out);
        CUDDH_CUDA_CHECK(cudaEventRecord(stop));
        CUDDH_CUDA_CHECK(cudaEventSynchronize(stop));
        CUDDH_CUDA_CHECK(cudaEventElapsedTime(&waveholtz_runtimes_ms[i], start, stop));

        CUDDH_CUDA_CHECK(cudaEventRecord(start));
        F_mr.action(in, mr_out);
        CUDDH_CUDA_CHECK(cudaEventRecord(stop));
        CUDDH_CUDA_CHECK(cudaEventSynchronize(stop));
        CUDDH_CUDA_CHECK(cudaEventElapsedTime(&minres_runtimes_ms[i], start, stop));
    }

    CUDDH_CUDA_CHECK(cudaEventDestroy(start));
    CUDDH_CUDA_CHECK(cudaEventDestroy(stop));

    const float waveholtz_total_ms = std::accumulate(waveholtz_runtimes_ms.begin(), waveholtz_runtimes_ms.end(), 0.0f);
    const float waveholtz_avg_ms = waveholtz_total_ms / static_cast<float>(iterations);
    const auto [waveholtz_min_it, waveholtz_max_it] =
        std::minmax_element(waveholtz_runtimes_ms.begin(), waveholtz_runtimes_ms.end());

    const float minres_total_ms = std::accumulate(minres_runtimes_ms.begin(), minres_runtimes_ms.end(), 0.0f);
    const float minres_avg_ms = minres_total_ms / static_cast<float>(iterations);
    const auto [minres_min_it, minres_max_it] =
        std::minmax_element(minres_runtimes_ms.begin(), minres_runtimes_ms.end());

    const float waveholtz_p10_ms = percentile_ms(waveholtz_runtimes_ms, 0.10);
    const float waveholtz_p50_ms = percentile_ms(waveholtz_runtimes_ms, 0.50);
    const float waveholtz_p90_ms = percentile_ms(waveholtz_runtimes_ms, 0.90);
    const float minres_p10_ms = percentile_ms(minres_runtimes_ms, 0.10);
    const float minres_p50_ms = percentile_ms(minres_runtimes_ms, 0.50);
    const float minres_p90_ms = percentile_ms(minres_runtimes_ms, 0.90);

    auto relative_to_waveholtz = [&](float t_ms) -> float {
        if (waveholtz_avg_ms <= 0.0f)
            return 0.0f;
        return t_ms / waveholtz_avg_ms;
    };

    auto print_time_with_relative = [&](const char *label, float t_ms) {
        std::cout << "  " << std::left << std::setw(24) << label << std::right << std::fixed << std::setprecision(3)
                  << t_ms << " [ms] " << std::setprecision(1) << "(x" << relative_to_waveholtz(t_ms) << ")\n";
    };

    std::cout << "Benchmarking DDSubstructuredOperator<" << precision
              << ">::action with WaveHoltz vs MINRES subdomain solvers\n"
              << "  precision:               " << precision << "\n"
              << "  degree:                  " << degree << "\n"
              << "  mesh:                    " << nx << " x " << ny << "\n"
              << "  omega (fixed):           " << omega << "\n"
              << "  subdomains dims:         " << sx << " x " << sy << "\n"
              << "  WaveHoltz kernel:        " << F_wh.kernel_str() << "\n"
              << "  MINRES kernel:           " << F_mr.kernel_str() << "\n"
              << "  #subdomains:             " << efem.size() << "\n"
              << "  #dof:                    " << 2 * fem.size() << "\n"
              << "  #lambda dof:             " << F_wh.size() << "\n"
              << "  warmup:                  " << warmup << "\n"
              << "  iterations:              " << iterations << "\n";

    std::cout << "  WaveHoltz timings (baseline):\n";
    print_time_with_relative("min time:", *waveholtz_min_it);
    print_time_with_relative("max time:", *waveholtz_max_it);
    print_time_with_relative("avg time:", waveholtz_avg_ms);
    print_time_with_relative("p10 time:", waveholtz_p10_ms);
    print_time_with_relative("p50 time:", waveholtz_p50_ms);
    print_time_with_relative("p90 time:", waveholtz_p90_ms);
    std::cout << std::fixed << std::setprecision(3) << "  " << std::left << std::setw(24) << "total time:" << std::right
              << waveholtz_total_ms << " [ms]\n";

    std::cout << "  MINRES timings:\n";
    print_time_with_relative("min time:", *minres_min_it);
    print_time_with_relative("max time:", *minres_max_it);
    print_time_with_relative("avg time:", minres_avg_ms);
    print_time_with_relative("p10 time:", minres_p10_ms);
    print_time_with_relative("p50 time:", minres_p50_ms);
    print_time_with_relative("p90 time:", minres_p90_ms);
    std::cout << std::fixed << std::setprecision(3) << "  " << std::left << std::setw(24) << "total time:" << std::right
              << minres_total_ms << " [ms]" << std::endl;

    if (!output_file.empty())
    {
        const bool ok = write_csv_row(
            output_file, precision, degree, nx, ny, omega, sx, sy, static_cast<int>(config.block_size), config.tdof,
            warmup, iterations, efem.size(), 2 * fem.size(), F_wh.size(), *waveholtz_min_it, *waveholtz_max_it,
            waveholtz_avg_ms, waveholtz_total_ms, waveholtz_p10_ms, waveholtz_p50_ms, waveholtz_p90_ms, *minres_min_it,
            *minres_max_it, minres_avg_ms, minres_total_ms, minres_p10_ms, minres_p50_ms, minres_p90_ms);
        if (!ok)
            throw std::runtime_error("failed to write output file: " + output_file);
    }

    return 0;
}

int main(int argc, char **argv)
{
    CLI::App app{"Benchmark DDSubstructuredOperator::action (WaveHoltz vs MINRES) in 2D"};

    std::string precision = "float";
    int degree = 3;
    std::vector<int> mesh_dims{64, 64};
    std::vector<int> subdomain_dims{8, 8};
    int block_size = 0;
    int tdof = 0;
    int warmup = 5;
    int iterations = 20;
    std::string output_file;

    app.add_option("-p,--precision", precision, "Scalar type: float or double");
    app.add_option("-d,--degree", degree, "Polynomial degree")->required();
    app.add_option("--mesh", mesh_dims, "Mesh dimensions: nx ny")->expected(2)->required();
    app.add_option("--subdomains", subdomain_dims, "Desired subdomain dimensions: sx sy")->expected(2)->required();
    app.add_option("--block-size", block_size, "Kernel block size: 0, 256, 512, or 1024");
    app.add_option("--tdof", tdof, "Kernel tdof value");
    app.add_option("--warmup", warmup, "Warmup action calls before timing");
    app.add_option("-n,--iterations", iterations, "Number of timed action calls");
    app.add_option("-o,--output", output_file,
                   "Optional CSV output path. Appends one row and writes a header automatically if needed.");

    CLI11_PARSE(app, argc, argv);

    const int nx = mesh_dims[0];
    const int ny = mesh_dims[1];
    const int sx = subdomain_dims[0];
    const int sy = subdomain_dims[1];

    if (degree < 1)
        throw std::runtime_error("--degree must be positive");
    if (nx < 1 || ny < 1)
        throw std::runtime_error("--mesh entries must be positive");
    if (sx < 1 || sy < 1)
        throw std::runtime_error("--subdomains entries must be positive");
    if (precision != "float" && precision != "double")
        throw std::runtime_error("--precision must be either 'float' or 'double'");
    if (tdof < 0 || tdof > 4)
        throw std::runtime_error("--tdof must be between 0 and 4");
    if (warmup < 0)
        throw std::runtime_error("--warmup must be non-negative");
    if (iterations < 1)
        throw std::runtime_error("--iterations must be positive");

    const DDKernelConfig config = {
        .block_size = parse_block_size(block_size),
        .tdof = tdof,
    };

    if (precision == "float")
        return run_benchmark<float>("float", degree, nx, ny, sx, sy, config, warmup, iterations, output_file);

    return run_benchmark<double>("double", degree, nx, ny, sx, sy, config, warmup, iterations, output_file);
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

static float percentile_ms(std::vector<float> samples, double q)
{
    if (samples.empty())
        return 0.0f;

    if (q <= 0.0)
        q = 0.0;
    else if (q >= 1.0)
        q = 1.0;

    std::sort(samples.begin(), samples.end());
    const double pos = q * static_cast<double>(samples.size() - 1);
    const int lo = static_cast<int>(pos);
    const int hi = std::min(lo + 1, static_cast<int>(samples.size() - 1));
    const double t = pos - static_cast<double>(lo);
    return static_cast<float>((1.0 - t) * static_cast<double>(samples[lo]) + t * static_cast<double>(samples[hi]));
}

static bool write_csv_row(const std::string &output_file, const std::string &precision, int degree, int nx, int ny,
                          double omega, int sx, int sy, int kernel_block_size, int kernel_tdof, int warmup,
                          int iterations, int n_subdomains, int n_dof, int n_lambda_dof, float waveholtz_min_ms,
                          float waveholtz_max_ms, float waveholtz_avg_ms, float waveholtz_total_ms,
                          float waveholtz_p10_ms, float waveholtz_p50_ms, float waveholtz_p90_ms, float minres_min_ms,
                          float minres_max_ms, float minres_avg_ms, float minres_total_ms, float minres_p10_ms,
                          float minres_p50_ms, float minres_p90_ms)
{
    std::ofstream out(output_file, std::ios::app);
    if (!out)
        return false;

    std::ifstream in(output_file);
    const bool empty_file = !in.good() || (in.peek() == std::ifstream::traits_type::eof());

    if (empty_file)
    {
        out << "precision,degree,nx,ny,omega,sx,sy,kernel_block_size,kernel_tdof,warmup,iterations,"
            << "n_subdomains,n_dof,n_lambda_dof,waveholtz_min_ms,waveholtz_max_ms,waveholtz_avg_ms,"
            << "waveholtz_total_ms,waveholtz_p10_ms,waveholtz_p50_ms,waveholtz_p90_ms,minres_min_ms,minres_max_ms,"
            << "minres_avg_ms,minres_total_ms,minres_p10_ms,minres_p50_ms,minres_p90_ms,"
            << "minres_min_rel_to_waveholtz,minres_max_rel_to_waveholtz,minres_avg_rel_to_waveholtz,"
            << "minres_p10_rel_to_waveholtz,minres_p50_rel_to_waveholtz,minres_p90_rel_to_waveholtz\n";
    }

    auto rel = [&](float t) -> float {
        if (waveholtz_avg_ms <= 0.0f)
            return 0.0f;
        return t / waveholtz_avg_ms;
    };

    out << precision << ',' << degree << ',' << nx << ',' << ny << ',' << omega << ',' << sx << ',' << sy << ','
        << kernel_block_size << ',' << kernel_tdof << ',' << warmup << ',' << iterations << ',' << n_subdomains << ','
        << n_dof << ',' << n_lambda_dof << ',' << waveholtz_min_ms << ',' << waveholtz_max_ms << ',' << waveholtz_avg_ms
        << ',' << waveholtz_total_ms << ',' << waveholtz_p10_ms << ',' << waveholtz_p50_ms << ',' << waveholtz_p90_ms
        << ',' << minres_min_ms << ',' << minres_max_ms << ',' << minres_avg_ms << ',' << minres_total_ms << ','
        << minres_p10_ms << ',' << minres_p50_ms << ',' << minres_p90_ms << ',' << rel(minres_min_ms) << ','
        << rel(minres_max_ms) << ',' << rel(minres_avg_ms) << ',' << rel(minres_p10_ms) << ',' << rel(minres_p50_ms)
        << ',' << rel(minres_p90_ms) << '\n';

    return static_cast<bool>(out);
}