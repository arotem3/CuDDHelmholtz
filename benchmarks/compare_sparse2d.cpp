/**
 * compare_sparse2d.cpp
 *
 * Two-mode benchmark comparing WaveHoltz vs SparseDirect subdomain solvers.
 * Requires CuDSS (-DCUDDH_USE_CUDSS=ON).
 *
 * Modes
 * ─────
 *   kernel  — Per-action timing (N repeated action() calls on a fixed random input).
 *             SparseDirect factorization cost is reported separately.
 *             Output CSV columns cover per-action p10/p50/p90 plus build stats.
 *
 *   e2e     — Full end-to-end DDH solve (build + MINRES outer solve) on a
 *             two-Gaussian Helmholtz forcing with piecewise-constant coefficient.
 *             Output CSV includes build time, solve time, iterations, final
 *             residual, factorization memory, and per-action timing from the
 *             SparseDirect batch LU stats.
 *
 * Each invocation appends exactly one row to --output.
 */

#include <algorithm>
#include <format>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "CLI11.hpp"
#include "cuddh.hpp"

using namespace cuddh;

// ─── Utility ─────────────────────────────────────────────────────────────────

static DDKernelConfig::BlockSize parse_block_size(int v)
{
    switch (v)
    {
        case 256:
            return DDKernelConfig::t256;
        case 512:
            return DDKernelConfig::t512;
        case 1024:
            return DDKernelConfig::t1024;
        default:
            return DDKernelConfig::Default;
    }
}

static float percentile_ms(std::vector<float> s, double q)
{
    if (s.empty())
        return 0.0f;
    q = std::clamp(q, 0.0, 1.0);
    std::sort(s.begin(), s.end());
    const double pos = q * (static_cast<double>(s.size()) - 1.0);
    const int lo = static_cast<int>(pos);
    const int hi = std::min(lo + 1, static_cast<int>(s.size()) - 1);
    const double t = pos - lo;
    return static_cast<float>((1.0 - t) * s[lo] + t * s[hi]);
}

// Append one row to a CSV file, writing a header line first if the file is empty.
static bool csv_append(const std::string &path, const std::string &header, const std::string &row)
{
    // Detect whether the file is empty before opening in append mode.
    std::ifstream probe(path, std::ios::ate);
    const bool needs_header = !probe || probe.tellg() == 0;
    probe.close();

    std::ofstream out(path, std::ios::app);
    if (!out)
        return false;
    if (needs_header)
        out << header << '\n';
    out << row << '\n';
    return static_cast<bool>(out);
}

// ─── Kernel mode ─────────────────────────────────────────────────────────────

template <typename scalar_t>
static int run_kernel(const char *precision, int degree, int nx, int ny, int sx, int sy, DDKernelConfig config,
                      int warmup, int iterations, const std::string &output_file)
{
#ifndef CUDDH_HAS_CUDSS
    (void)precision;
    (void)degree;
    (void)nx;
    (void)ny;
    (void)sx;
    (void)sy;
    (void)config;
    (void)warmup;
    (void)iterations;
    (void)output_file;
    std::cerr << "compare_sparse2d: kernel mode requires CuDSS. Rebuild with -DCUDDH_USE_CUDSS=ON.\n";
    return 1;
#else
    std::mt19937 gen(42);
    std::uniform_real_distribution<scalar_t> dist(-1.0f, 1.0f);

    const double omega = 0.1 * static_cast<double>(std::max(nx, ny)) * static_cast<double>(degree);

    Mesh2D mesh = Mesh2D::uniform_rect(nx, -1.0, 1.0, ny, -1.0, 1.0);
    Basis basis(degree + 1);
    H1Space2D fem(mesh, basis);
    EnsembleSpace efem = partition_uniform_rect(fem, {nx, ny}, {sx, sy});

    auto a = gridfunc(fem, [] __device__(const double2) -> double { return 1.0; });

    // Build WaveHoltz operator (construction is negligible; no factorization).
    DDSubstructuredOperator<scalar_t, SubdomainSolver::WaveHoltz> F_wh(efem, omega, a, config);
    CUDDH_CUDA_CHECK(cudaDeviceSynchronize());

    // Build SparseDirect operator — construction includes symbolic analysis + LU factorization.
    Timer sd_build_timer;
    DDSubstructuredOperator<scalar_t, SubdomainSolver::SparseDirect> F_sd(efem, omega, a);
    CUDDH_CUDA_CHECK(cudaDeviceSynchronize());
    const double sd_build_s = sd_build_timer.elapsed();

    const int n_lambda = F_wh.ndof();

    thrust::universal_vector<scalar_t> lambda_in(n_lambda);
    thrust::universal_vector<scalar_t> wh_out(n_lambda);
    thrust::universal_vector<scalar_t> sd_out(n_lambda);
    for (auto &v : lambda_in)
        v = dist(gen);

    scalar_t *in = thrust::raw_pointer_cast(lambda_in.data());
    scalar_t *wo = thrust::raw_pointer_cast(wh_out.data());
    scalar_t *so = thrust::raw_pointer_cast(sd_out.data());

    // Warmup — excluded from timing.
    for (int i = 0; i < warmup; ++i)
    {
        F_wh.action(in, wo);
        F_sd.action(in, so);
        CUDDH_CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Timed runs with CUDA events for sub-millisecond precision.
    cudaEvent_t ev_start, ev_stop;
    CUDDH_CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDDH_CUDA_CHECK(cudaEventCreate(&ev_stop));

    std::vector<float> wh_ms(iterations), sd_ms(iterations);
    for (int i = 0; i < iterations; ++i)
    {
        CUDDH_CUDA_CHECK(cudaEventRecord(ev_start));
        F_wh.action(in, wo);
        CUDDH_CUDA_CHECK(cudaEventRecord(ev_stop));
        CUDDH_CUDA_CHECK(cudaEventSynchronize(ev_stop));
        CUDDH_CUDA_CHECK(cudaEventElapsedTime(&wh_ms[i], ev_start, ev_stop));

        CUDDH_CUDA_CHECK(cudaEventRecord(ev_start));
        F_sd.action(in, so);
        CUDDH_CUDA_CHECK(cudaEventRecord(ev_stop));
        CUDDH_CUDA_CHECK(cudaEventSynchronize(ev_stop));
        CUDDH_CUDA_CHECK(cudaEventElapsedTime(&sd_ms[i], ev_start, ev_stop));
    }

    CUDDH_CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDDH_CUDA_CHECK(cudaEventDestroy(ev_stop));

    // WaveHoltz statistics.
    const float wh_total = std::accumulate(wh_ms.begin(), wh_ms.end(), 0.0f);
    const float wh_avg = wh_total / static_cast<float>(iterations);
    const auto [wh_min_it, wh_max_it] = std::minmax_element(wh_ms.begin(), wh_ms.end());
    const float wh_p10 = percentile_ms(wh_ms, 0.10);
    const float wh_p50 = percentile_ms(wh_ms, 0.50);
    const float wh_p90 = percentile_ms(wh_ms, 0.90);

    // SparseDirect statistics.
    const float sd_total = std::accumulate(sd_ms.begin(), sd_ms.end(), 0.0f);
    const float sd_avg = sd_total / static_cast<float>(iterations);
    const auto [sd_min_it, sd_max_it] = std::minmax_element(sd_ms.begin(), sd_ms.end());
    const float sd_p10 = percentile_ms(sd_ms, 0.10);
    const float sd_p50 = percentile_ms(sd_ms, 0.50);
    const float sd_p90 = percentile_ms(sd_ms, 0.90);

    const auto &sd_lu = F_sd.block_lu().stats();

    // Console output.
    std::cout << std::format("compare_sparse2d [kernel]  precision={}, degree={}, mesh={}x{}, sub={}x{}\n", precision,
                             degree, nx, ny, sx, sy)
              << std::format("  omega={:.2f}, n_subdomains={}, n_dof={}, n_lambda={}\n", omega, efem.size(),
                             2 * fem.size(), n_lambda)
              << std::format("  WaveHoltz kernel:   {}\n", F_wh.kernel_str())
              << std::format("  WaveHoltz action:   p50={:.3f} ms, p10={:.3f} ms, p90={:.3f} ms, avg={:.3f} ms\n",
                             wh_p50, wh_p10, wh_p90, wh_avg)
              << std::format("  SparseDirect build: analysis={:.3e} s, factor={:.3e} s, memory={:.3f} MiB\n",
                             sd_lu.analysis_seconds, sd_lu.factor_seconds, sd_lu.factor_mib)
              << std::format("  SparseDirect build: total wall-clock = {:.3e} s\n", sd_build_s)
              << std::format("  SparseDirect action: p50={:.3f} ms, p10={:.3f} ms, p90={:.3f} ms, avg={:.3f} ms\n",
                             sd_p50, sd_p10, sd_p90, sd_avg);

    if (!output_file.empty())
    {
        const std::string header =
            "precision,degree,nx,ny,omega,sx,sy,kernel_block_size,kernel_tdof,warmup,iterations,"
            "n_subdomains,n_dof,n_lambda_dof,"
            "wh_min_ms,wh_max_ms,wh_avg_ms,wh_p10_ms,wh_p50_ms,wh_p90_ms,wh_total_ms,"
            "sd_analysis_s,sd_factor_s,sd_build_s,sd_factor_mib,"
            "sd_min_ms,sd_max_ms,sd_avg_ms,sd_p10_ms,sd_p50_ms,sd_p90_ms,sd_total_ms";

        const std::string row = std::format(
            "{},{},{},{},{:.6g},{},{},{},{},{},{},"
            "{},{},{},"
            "{},{},{},{},{},{},{},"
            "{:.6e},{:.6e},{:.6e},{:.6f},"
            "{},{},{},{},{},{},{}",
            precision, degree, nx, ny, omega, sx, sy, static_cast<int>(config.block_size), config.tdof, warmup,
            iterations, efem.size(), 2 * fem.size(), n_lambda, *wh_min_it, *wh_max_it, wh_avg, wh_p10, wh_p50, wh_p90,
            wh_total, sd_lu.analysis_seconds, sd_lu.factor_seconds, sd_build_s, sd_lu.factor_mib, *sd_min_it,
            *sd_max_it, sd_avg, sd_p10, sd_p50, sd_p90, sd_total);

        if (!csv_append(output_file, header, row))
            throw std::runtime_error("failed to write output: " + output_file);
    }

    return 0;
#endif
}

// ─── E2E mode ────────────────────────────────────────────────────────────────

template <typename scalar_t>
static int run_e2e(const char *precision, int degree, int nx, int ny, int sx, int sy, DDKernelConfig config, int maxit,
                   double rtol, const std::string &output_file)
{
#ifndef CUDDH_HAS_CUDSS
    (void)precision;
    (void)degree;
    (void)nx;
    (void)ny;
    (void)sx;
    (void)sy;
    (void)config;
    (void)maxit;
    (void)rtol;
    (void)output_file;
    std::cerr << "compare_sparse2d: e2e mode requires CuDSS. Rebuild with -DCUDDH_USE_CUDSS=ON.\n";
    return 1;
#else
    const double omega = 0.1 * static_cast<double>(std::max(nx, ny)) * static_cast<double>(degree);

    Mesh2D mesh = Mesh2D::uniform_rect(nx, -1.0, 1.0, ny, -1.0, 1.0);
    Basis basis(degree + 1);
    H1Space2D fem(mesh, basis);
    EnsembleSpace efem = partition_uniform_rect(fem, {nx, ny}, {sx, sy});
    const int N = 2 * fem.size();

    auto a = gridfunc(
        fem, [] __device__(const double2 X) -> double { return (max(fabs(X.x), fabs(X.y)) < 0.5) ? 0.5 : 1.0; });

    const int n_lambda = [&]() -> int {
        DDSubstructuredOperator<scalar_t, SubdomainSolver::WaveHoltz> tmp(efem, omega, a, config);
        return tmp.ndof();
    }();

    // ── Build WaveHoltz solver ────────────────────────────────────────────────
    Timer wh_build_timer;
    DDH<scalar_t, SubdomainSolver::WaveHoltz> wh_solver(efem, omega, a, config);
    CUDDH_CUDA_CHECK(cudaDeviceSynchronize());
    const double wh_build_s = wh_build_timer.elapsed();

    // ── Build SparseDirect solver (includes analysis + factorization) ─────────
    Timer sd_build_timer;
    DDH<scalar_t, SubdomainSolver::SparseDirect> sd_solver(efem, omega, a);
    CUDDH_CUDA_CHECK(cudaDeviceSynchronize());
    const double sd_build_s = sd_build_timer.elapsed();

    // ── RHS ──────────────────────────────────────────────────────────────────
    thrust::universal_vector<double> B(N, 0.0);
    double *b = thrust::raw_pointer_cast(B.data());
    l2_project(b, MassMatrix(fem), [omega] __device__(const double2 X) -> double {
        const double s = omega * omega;
        double r = (X.x + 0.5) * (X.x + 0.5) + X.y * X.y;
        double F = s / M_PI * exp(-s * r);
        r = (X.x - 0.5) * (X.x - 0.5) + (X.y + 0.5) * (X.y + 0.5);
        return F + s / M_PI * exp(-s * r);
    });

    const SolverParams opts = {.maxit = maxit, .rtol = rtol, .verbose = SolverParams::Silent};

    // ── WaveHoltz solve ───────────────────────────────────────────────────────
    thrust::universal_vector<double> U_wh(N, 0.0);
    double *u_wh = thrust::raw_pointer_cast(U_wh.data());

    Timer wh_solve_timer;
    const SolverResults wh_out = wh_solver.solve(u_wh, b, opts);
    CUDDH_CUDA_CHECK(cudaDeviceSynchronize());
    const double wh_solve_s = wh_solve_timer.elapsed();

    // ── SparseDirect solve ────────────────────────────────────────────────────
    thrust::universal_vector<double> U_sd(N, 0.0);
    double *u_sd = thrust::raw_pointer_cast(U_sd.data());

    Timer sd_solve_timer;
    const SolverResults sd_out = sd_solver.solve(u_sd, b, opts);
    CUDDH_CUDA_CHECK(cudaDeviceSynchronize());
    const double sd_solve_s = sd_solve_timer.elapsed();

    // ── Residuals |b - A u| / |b| ─────────────────────────────────────────────
    auto compute_residual = [&](const double *u_vec) -> double {
        ivec boundary_faces = mesh.boundary_edges();
        TraceSpace2D fs(fem, boundary_faces.size(), boundary_faces);
        Helmholtz A(fem, fs, omega, a);
        thrust::universal_vector<double> Au(N, 0.0);
        A.action(u_vec, thrust::raw_pointer_cast(Au.data()));
        return dla::dist(N, thrust::raw_pointer_cast(Au.data()), b) / dla::norm(N, b);
    };
    const double wh_residual = compute_residual(u_wh);
    const double sd_residual = compute_residual(u_sd);

    // ── SparseDirect per-action stats ─────────────────────────────────────────
    // solve_seconds has one entry per outer MINRES iteration.
    const auto &sd_lu_stats = sd_solver.op().block_lu().stats();
    const auto &sv = sd_lu_stats.solve_seconds;
    const double sd_action_min_ms = sv.empty() ? 0.0 : *std::min_element(sv.begin(), sv.end()) * 1000.0;
    const double sd_action_avg_ms =
        sv.empty() ? 0.0 : std::accumulate(sv.begin(), sv.end(), 0.0) / static_cast<double>(sv.size()) * 1000.0;
    const double sd_action_max_ms = sv.empty() ? 0.0 : *std::max_element(sv.begin(), sv.end()) * 1000.0;

    const double wh_total_s = wh_build_s + wh_solve_s;
    const double sd_total_s = sd_build_s + sd_solve_s;

    // ── Console output ────────────────────────────────────────────────────────
    std::cout << std::format("compare_sparse2d [e2e]  precision={}, degree={}, mesh={}x{}, sub={}x{}\n", precision,
                             degree, nx, ny, sx, sy)
              << std::format("  omega={:.2f}, n_subdomains={}, n_dof={}, n_lambda={}\n", omega, efem.size(), N,
                             n_lambda)
              << std::format(
                     "  WaveHoltz:    build={:.3e} s, solve={:.3e} s, total={:.3e} s,"
                     " iters={}, residual={:.2e}\n",
                     wh_build_s, wh_solve_s, wh_total_s, wh_out.num_iter, wh_residual)
              << std::format(
                     "  SparseDirect: build={:.3e} s, solve={:.3e} s, total={:.3e} s,"
                     " iters={}, residual={:.2e}\n",
                     sd_build_s, sd_solve_s, sd_total_s, sd_out.num_iter, sd_residual)
              << std::format("    LU: analysis={:.3e} s, factor={:.3e} s, factor_mib={:.3f}\n",
                             sd_lu_stats.analysis_seconds, sd_lu_stats.factor_seconds, sd_lu_stats.factor_mib)
              << std::format("    per-action: min={:.3f} ms, avg={:.3f} ms, max={:.3f} ms\n", sd_action_min_ms,
                             sd_action_avg_ms, sd_action_max_ms);

    if (!output_file.empty())
    {
        const std::string header =
            "precision,degree,nx,ny,omega,sx,sy,kernel_block_size,kernel_tdof,maxit,rtol,"
            "n_subdomains,n_dof,n_lambda_dof,"
            "wh_build_s,wh_solve_s,wh_total_s,wh_iterations,wh_residual,"
            "sd_build_s,sd_solve_s,sd_total_s,sd_iterations,sd_residual,"
            "sd_analysis_s,sd_factor_s,sd_factor_mib,"
            "sd_action_min_ms,sd_action_avg_ms,sd_action_max_ms";

        const std::string row = std::format(
            "{},{},{},{},{:.6g},{},{},{},{},{},{:.6g},"
            "{},{},{},"
            "{:.6e},{:.6e},{:.6e},{},{:.6e},"
            "{:.6e},{:.6e},{:.6e},{},{:.6e},"
            "{:.6e},{:.6e},{:.6f},"
            "{:.6f},{:.6f},{:.6f}",
            precision, degree, nx, ny, omega, sx, sy, static_cast<int>(config.block_size), config.tdof, maxit, rtol,
            efem.size(), N, n_lambda, wh_build_s, wh_solve_s, wh_total_s, wh_out.num_iter, wh_residual, sd_build_s,
            sd_solve_s, sd_total_s, sd_out.num_iter, sd_residual, sd_lu_stats.analysis_seconds,
            sd_lu_stats.factor_seconds, sd_lu_stats.factor_mib, sd_action_min_ms, sd_action_avg_ms, sd_action_max_ms);

        if (!csv_append(output_file, header, row))
            throw std::runtime_error("failed to write output: " + output_file);
    }

    return 0;
#endif
}

// ─── Main ────────────────────────────────────────────────────────────────────

int main(int argc, char **argv)
{
    std::string mode = "kernel";
    std::string precision = "float";
    int degree = 3;
    std::vector<int> mesh_dims{64, 64};
    std::vector<int> subdomain_dims{8, 8};
    int block_size = 0;
    int tdof = 0;
    int warmup = 5;
    int iterations = 20;
    int maxit = 1000;
    double rtol = 1e-3;
    std::string output_file;

    CLI::App app{"Benchmark WaveHoltz vs SparseDirect in 2D (kernel action timing and end-to-end solve)"};
    app.add_option("--mode", mode, "Benchmark mode: kernel | e2e")
        ->default_val("kernel")
        ->check(CLI::IsMember({"kernel", "e2e"}, CLI::ignore_case));
    app.add_option("-p,--precision", precision, "Scalar type: float or double")
        ->default_val("float")
        ->check(CLI::IsMember({"float", "double"}));
    app.add_option("-d,--degree", degree, "Polynomial degree")->default_val(3);
    app.add_option("--mesh", mesh_dims, "Mesh dimensions: nx ny")->expected(2)->required();
    app.add_option("--subdomains", subdomain_dims, "Subdomain element dims: sx sy")->expected(2)->required();
    app.add_option("--block-size", block_size, "WaveHoltz kernel block size: 0 (auto), 256, 512, 1024")
        ->default_val(0)
        ->check(CLI::IsMember({0, 256, 512, 1024}));
    app.add_option("--tdof", tdof, "WaveHoltz kernel DOFs/thread: 0 (auto), 1..4")
        ->default_val(0)
        ->check(CLI::Range(0, 4));
    // kernel-mode options
    app.add_option("--warmup", warmup, "[kernel] Warmup action calls before timing")->default_val(5);
    app.add_option("-n,--iterations", iterations, "[kernel] Number of timed action calls")->default_val(20);
    // e2e-mode options
    app.add_option("--maxit", maxit, "[e2e] Maximum outer MINRES iterations")->default_val(1000);
    app.add_option("--rtol", rtol, "[e2e] Relative tolerance for outer MINRES")->default_val(1e-3);
    app.add_option("-o,--output", output_file, "CSV output path (appends one row; writes header if file is empty)");

    CLI11_PARSE(app, argc, argv);

    const int nx = mesh_dims[0];
    const int ny = mesh_dims[1];
    const int sx = subdomain_dims[0];
    const int sy = subdomain_dims[1];

    if (degree < 1)
        throw std::runtime_error("--degree must be positive");
    if (nx < 1 || ny < 1 || sx < 1 || sy < 1)
        throw std::runtime_error("mesh and subdomain dimensions must be positive");
    if (warmup < 0)
        throw std::runtime_error("--warmup must be non-negative");
    if (iterations < 1)
        throw std::runtime_error("--iterations must be positive");

    const DDKernelConfig config = {.block_size = parse_block_size(block_size), .tdof = tdof};

    const bool e2e = (CLI::detail::to_lower(mode) == "e2e");

    if (precision == "float")
        return e2e ? run_e2e<float>("float", degree, nx, ny, sx, sy, config, maxit, rtol, output_file)
                   : run_kernel<float>("float", degree, nx, ny, sx, sy, config, warmup, iterations, output_file);

    return e2e ? run_e2e<double>("double", degree, nx, ny, sx, sy, config, maxit, rtol, output_file)
               : run_kernel<double>("double", degree, nx, ny, sx, sy, config, warmup, iterations, output_file);
}
