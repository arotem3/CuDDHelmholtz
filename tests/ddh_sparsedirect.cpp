#include "test_common.hpp"

#ifdef CUDDH_HAS_CUDSS

#include <chrono>

using namespace cuddh;

// Compare SparseDirect action against MINRES action on the same random lambda.
// MINRES subdomain solves converge to ~1e-12 (double) / ~1e-6 (float), so
// SparseDirect (exact) should match within those bounds.
template <typename scalar_t>
static void test_operator_consistency(TestLogger &summary, std::string_view precision, double pass_rtol)
{
    constexpr int N = 32;
    constexpr int DEG = 2;
    const double omega = 3.0;

    std::mt19937 gen(42);
    std::uniform_real_distribution<scalar_t> dist(-1, 1);

    Mesh2D mesh = Mesh2D::uniform_rect(N, -1.0, 1.0, N, -1.0, 1.0);
    Basis basis(DEG + 1);
    H1Space2D fem(mesh, basis);
    EnsembleSpace efem = partition_uniform_rect(fem, {N, N}, {8, 8});

    GridFunc2D<double> a(fem);
    {
        auto h_a = a.write(MemorySpace::HOST);
        std::fill(h_a.begin(), h_a.end(), 1.5);
    }

    DDSubstructuredOperator<scalar_t, SubdomainSolver::MINRES> F_mr(efem, omega, a);
    DDSubstructuredOperator<scalar_t, SubdomainSolver::SparseDirect> F_sd(efem, omega, a);

    const int n = F_mr.ndof();
    thrust::universal_vector<scalar_t> lambda(n);
    for (auto &v : lambda)
        v = dist(gen);
    const scalar_t *x = thrust::raw_pointer_cast(lambda.data());

    thrust::device_vector<scalar_t> y_mr(n, scalar_t(0));
    thrust::device_vector<scalar_t> y_sd(n, scalar_t(0));
    F_mr.action(x, thrust::raw_pointer_cast(y_mr.data()));
    F_sd.action(x, thrust::raw_pointer_cast(y_sd.data()));
    CUDDH_CUDA_CHECK(cudaDeviceSynchronize());

    const scalar_t *pm = thrust::raw_pointer_cast(y_mr.data());
    const scalar_t *ps = thrust::raw_pointer_cast(y_sd.data());
    const scalar_t ref_norm = dla::norm(n, pm);
    const scalar_t diff_norm = dla::dist(n, pm, ps);
    const double rel_diff = (ref_norm > 0) ? double(diff_norm / ref_norm) : double(diff_norm);

    std::cout << std::format("  [info] Operator consistency {}: rel_diff = {:.2e}, tolerance = {:.2e}\n", precision,
                             rel_diff, pass_rtol);

    const auto name = std::format("Operator consistency MINRES vs SparseDirect ({})", precision);
    if (rel_diff <= pass_rtol)
        summary.pass(name);
    else
        summary.fail(name, std::format("rel_diff = {:.2e}, tolerance = {:.2e}", rel_diff, pass_rtol));
}

// SparseDirect and WaveHoltz share the same rhs() implementation (it doesn't
// depend on the subdomain solver), so results should match to floating-point rounding.
template <typename scalar_t>
static void test_rhs_consistency(TestLogger &summary, std::string_view precision, double pass_rtol)
{
    constexpr int N = 32;
    constexpr int DEG = 3;
    const double omega = 0.1 * N * DEG;

    std::mt19937 gen(42);
    std::uniform_real_distribution<scalar_t> dist(-1, 1);

    Mesh2D mesh = Mesh2D::uniform_rect(N, 0.0, 1.0, N, 0.0, 1.0);
    Basis basis(DEG + 1);
    H1Space2D fem(mesh, basis);
    EnsembleSpace efem = partition_uniform_rect(fem, {N, N}, {8, 8});

    GridFunc2D<double> a(fem);
    {
        auto h_a = a.write(MemorySpace::HOST);
        std::fill(h_a.begin(), h_a.end(), 1.5);
    }

    DDSubstructuredOperator<scalar_t, SubdomainSolver::WaveHoltz> F_wh(efem, omega, a);
    DDSubstructuredOperator<scalar_t, SubdomainSolver::SparseDirect> F_sd(efem, omega, a);

    const int n_dof = fem.size();
    const int n_lambda = F_wh.ndof();

    thrust::universal_vector<double> f(2 * n_dof);
    for (auto &v : f)
        v = dist(gen);
    const double *f_ptr = thrust::raw_pointer_cast(f.data());

    thrust::device_vector<scalar_t> rhs_wh(n_lambda, scalar_t(0));
    thrust::device_vector<scalar_t> rhs_sd(n_lambda, scalar_t(0));
    F_wh.rhs(f_ptr, thrust::raw_pointer_cast(rhs_wh.data()));
    F_sd.rhs(f_ptr, thrust::raw_pointer_cast(rhs_sd.data()));
    CUDDH_CUDA_CHECK(cudaDeviceSynchronize());

    const scalar_t *pw = thrust::raw_pointer_cast(rhs_wh.data());
    const scalar_t *ps = thrust::raw_pointer_cast(rhs_sd.data());
    const scalar_t ref_norm = dla::norm(n_lambda, pw);
    const scalar_t diff_norm = dla::dist(n_lambda, pw, ps);
    const double rel_diff = (ref_norm > 0) ? double(diff_norm / ref_norm) : double(diff_norm);

    std::cout << std::format("  [info] RHS consistency {}: rel_diff = {:.2e}, tolerance = {:.2e}\n", precision,
                             rel_diff, pass_rtol);

    const auto name = std::format("RHS consistency WaveHoltz vs SparseDirect ({})", precision);
    if (rel_diff <= pass_rtol)
        summary.pass(name);
    else
        summary.fail(name, std::format("rel_diff = {:.2e}, tolerance = {:.2e}", rel_diff, pass_rtol));
}

// postprocess() doesn't invoke the subdomain solver, so SparseDirect and
// WaveHoltz must agree to floating-point rounding.
template <typename scalar_t>
static void test_postprocessing_consistency(TestLogger &summary, std::string_view precision, double pass_rtol)
{
    constexpr int N = 32;
    constexpr int DEG = 3;
    const double omega = 0.1 * N * DEG;

    std::mt19937 gen(123);
    std::uniform_real_distribution<scalar_t> dist(-1, 1);

    Mesh2D mesh = Mesh2D::uniform_rect(N, 0.0, 1.0, N, 0.0, 1.0);
    Basis basis(DEG + 1);
    H1Space2D fem(mesh, basis);
    EnsembleSpace efem = partition_uniform_rect(fem, {N, N}, {8, 8});

    GridFunc2D<double> a(fem);
    {
        auto h_a = a.write(MemorySpace::HOST);
        std::fill(h_a.begin(), h_a.end(), 1.5);
    }

    DDSubstructuredOperator<scalar_t, SubdomainSolver::WaveHoltz> F_wh(efem, omega, a);
    DDSubstructuredOperator<scalar_t, SubdomainSolver::SparseDirect> F_sd(efem, omega, a);

    const int n_dof = fem.size();
    const int n_lambda = F_wh.ndof();

    thrust::universal_vector<double> f(2 * n_dof);
    for (auto &v : f)
        v = dist(gen);

    thrust::universal_vector<scalar_t> lambda(n_lambda);
    for (auto &v : lambda)
        v = dist(gen);

    const double *f_ptr = thrust::raw_pointer_cast(f.data());
    const scalar_t *l_ptr = thrust::raw_pointer_cast(lambda.data());

    thrust::device_vector<double> u_wh(2 * n_dof, 0.0);
    thrust::device_vector<double> u_sd(2 * n_dof, 0.0);
    F_wh.postprocess(l_ptr, f_ptr, thrust::raw_pointer_cast(u_wh.data()));
    F_sd.postprocess(l_ptr, f_ptr, thrust::raw_pointer_cast(u_sd.data()));
    CUDDH_CUDA_CHECK(cudaDeviceSynchronize());

    const double *uw = thrust::raw_pointer_cast(u_wh.data());
    const double *us = thrust::raw_pointer_cast(u_sd.data());
    const double ref_norm = dla::norm(2 * n_dof, uw);
    const double diff_norm = dla::dist(2 * n_dof, uw, us);
    const double rel_diff = (ref_norm > 0) ? diff_norm / ref_norm : diff_norm;

    std::cout << std::format("  [info] Postprocessing consistency {}: rel_diff = {:.2e}, tolerance = {:.2e}\n",
                             precision, rel_diff, pass_rtol);

    const auto name = std::format("Postprocessing consistency WaveHoltz vs SparseDirect ({})", precision);
    if (rel_diff <= pass_rtol)
        summary.pass(name);
    else
        summary.fail(name, std::format("rel_diff = {:.2e}, tolerance = {:.2e}", rel_diff, pass_rtol));
}

// Print per-action timing (informational — always passes).
// Reports average over `n_reps` action calls after one warmup.
template <typename scalar_t>
static void test_action_timing(TestLogger &summary, std::string_view precision)
{
    constexpr int N = 32;
    constexpr int DEG = 3;
    const double omega = 0.1 * N * DEG;
    constexpr int n_reps = 5;

    Mesh2D mesh = Mesh2D::uniform_rect(N, -1.0, 1.0, N, -1.0, 1.0);
    Basis basis(DEG + 1);
    H1Space2D fem(mesh, basis);
    EnsembleSpace efem = partition_uniform_rect(fem, {N, N}, {8, 8});

    GridFunc2D<double> a(fem);
    {
        auto h_a = a.write(MemorySpace::HOST);
        std::fill(h_a.begin(), h_a.end(), 1.0);
    }

    DDSubstructuredOperator<scalar_t, SubdomainSolver::SparseDirect> F_sd(efem, omega, a);
    const int n = F_sd.ndof();

    thrust::universal_vector<scalar_t> lambda(n, scalar_t(1));
    thrust::device_vector<scalar_t> y(n, scalar_t(0));
    const scalar_t *x = thrust::raw_pointer_cast(lambda.data());
    scalar_t *yp = thrust::raw_pointer_cast(y.data());

    // warmup
    F_sd.action(x, yp);
    CUDDH_CUDA_CHECK(cudaDeviceSynchronize());

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < n_reps; ++i)
        F_sd.action(x, yp);
    CUDDH_CUDA_CHECK(cudaDeviceSynchronize());
    double elapsed = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count();

    const auto name = std::format("SparseDirect action timing ({})", precision);
    std::cout << std::format("  [info] {} avg {:.2e} s ({} reps)\n", name, elapsed / n_reps, n_reps);
    summary.pass(name);
}

#endif // CUDDH_HAS_CUDSS

int main()
{
#ifdef CUDDH_HAS_CUDSS
    TestLogger summary;

    test_operator_consistency<double>(summary, "double", 1e-8);
    test_operator_consistency<float>(summary, "float", 1e-3);

    test_rhs_consistency<double>(summary, "double", 1e-10);
    test_rhs_consistency<float>(summary, "float", 1e-4);

    test_postprocessing_consistency<double>(summary, "double", 1e-10);
    test_postprocessing_consistency<float>(summary, "float", 1e-4);

    test_action_timing<double>(summary, "double");
    test_action_timing<float>(summary, "float");

    return summary.finish();
#else
    std::cout << "[SKIP] ddh_sparsedirect: compiled without CuDSS support\n";
    return 0;
#endif
}
