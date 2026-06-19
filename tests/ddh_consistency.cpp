#include "test_common.hpp"

using namespace cuddh;

// Check that the substructured operator with MINRES gives the same output as
// the WaveHoltz variant on the same random input vector.
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

    const scalar_t a_val = static_cast<scalar_t>(1.5);
    GridFunc2D<double> a(fem);
    auto h_a = a.write(MemorySpace::HOST);
    std::fill(h_a.begin(), h_a.end(), a_val);

    DDSubstructuredOperator<scalar_t, SubdomainSolver::WaveHoltz> F_wh(efem, omega, a);
    DDSubstructuredOperator<scalar_t, SubdomainSolver::MINRES> F_mr(efem, omega, a);

    const int n = F_wh.ndof();

    thrust::universal_vector<scalar_t> lambda(n);
    std::srand(42);
    for (auto &value : lambda)
        value = dist(gen);
    const scalar_t *x = thrust::raw_pointer_cast(lambda.data());

    thrust::device_vector<scalar_t> y_wh(n, 0);
    thrust::device_vector<scalar_t> y_mr(n, 0);
    F_wh.action(x, thrust::raw_pointer_cast(y_wh.data()));
    F_mr.action(x, thrust::raw_pointer_cast(y_mr.data()));
    CUDDH_CUDA_CHECK(cudaDeviceSynchronize());

    const scalar_t *pw = thrust::raw_pointer_cast(y_wh.data());
    const scalar_t *pm = thrust::raw_pointer_cast(y_mr.data());
    const scalar_t ref_norm = dla::norm(n, pw);
    const scalar_t diff_norm = dla::dist(n, pw, pm);
    const double rel_diff = (ref_norm > 0) ? double(diff_norm / ref_norm) : double(diff_norm);

    const auto test_name = std::format("Operator consistency WaveHoltz vs MINRES ({})", precision);
    if (rel_diff <= pass_rtol)
        summary.pass(test_name);
    else
        summary.fail(test_name, std::format("rel_diff = {:.2e}, tolerance = {:.2e}", rel_diff, pass_rtol));
}

// Test that RHS evaluation is consistent between solvers
template <typename scalar_t>
static void test_rhs_evaluation(TestLogger &summary, std::string_view precision, double pass_rtol)
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

    const scalar_t a_val = static_cast<scalar_t>(1.5);
    GridFunc2D<double> a(fem);
    auto h_a = a.write(MemorySpace::HOST);
    std::fill(h_a.begin(), h_a.end(), a_val);

    DDSubstructuredOperator<scalar_t, SubdomainSolver::WaveHoltz> F_wh(efem, omega, a);
    DDSubstructuredOperator<scalar_t, SubdomainSolver::MINRES> F_mr(efem, omega, a);

    int n_dof = fem.size();
    int n_lambda = F_wh.ndof();

    thrust::universal_vector<double> f(2 * n_dof, 0.0);
    for (auto &value : f)
        value = dist(gen);
    const double *f_ptr = thrust::raw_pointer_cast(f.data());

    thrust::device_vector<scalar_t> rhs_wh(n_lambda, scalar_t(0));
    thrust::device_vector<scalar_t> rhs_mr(n_lambda, scalar_t(0));

    F_wh.rhs(f_ptr, thrust::raw_pointer_cast(rhs_wh.data()));
    F_mr.rhs(f_ptr, thrust::raw_pointer_cast(rhs_mr.data()));
    CUDDH_CUDA_CHECK(cudaDeviceSynchronize());

    const scalar_t *pw = thrust::raw_pointer_cast(rhs_wh.data());
    const scalar_t *pm = thrust::raw_pointer_cast(rhs_mr.data());
    const scalar_t rhs_ref = dla::norm(n_lambda, pw);
    const scalar_t rhs_diff = dla::dist(n_lambda, pw, pm);
    const double rhs_rel = (rhs_ref > scalar_t(0)) ? double(rhs_diff / rhs_ref) : double(rhs_diff);

    const auto test_name = std::format("RHS evaluation consistency ({})", precision);
    if (rhs_rel <= pass_rtol)
        summary.pass(test_name);
    else
        summary.fail(test_name, std::format("rel_diff = {:.2e}, tolerance = {:.2e}", rhs_rel, pass_rtol));
}

// Test that postprocessing is consistent between solvers
template <typename scalar_t>
static void test_postprocessing(TestLogger &summary, std::string_view precision, double pass_rtol)
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

    const scalar_t a_val = static_cast<scalar_t>(1.5);
    GridFunc2D<double> a(fem);
    auto h_a = a.write(MemorySpace::HOST);
    std::fill(h_a.begin(), h_a.end(), a_val);

    DDSubstructuredOperator<scalar_t, SubdomainSolver::WaveHoltz> F_wh(efem, omega, a);
    DDSubstructuredOperator<scalar_t, SubdomainSolver::MINRES> F_mr(efem, omega, a);

    int n_dof = fem.size();
    int n_lambda = F_wh.ndof();

    thrust::universal_vector<double> f(2 * n_dof);
    for (auto &value : f)
        value = dist(gen);
    const double *f_ptr = thrust::raw_pointer_cast(f.data());

    thrust::universal_vector<scalar_t> lambda(n_lambda, scalar_t(0));
    for (auto &value : lambda)
        value = dist(gen);
    const scalar_t *lambda_ptr = thrust::raw_pointer_cast(lambda.data());

    thrust::device_vector<double> u_wh(2 * n_dof, 0.0);
    thrust::device_vector<double> u_mr(2 * n_dof, 0.0);

    F_wh.postprocess(lambda_ptr, f_ptr, thrust::raw_pointer_cast(u_wh.data()));
    F_mr.postprocess(lambda_ptr, f_ptr, thrust::raw_pointer_cast(u_mr.data()));
    CUDDH_CUDA_CHECK(cudaDeviceSynchronize());

    const double *uw = thrust::raw_pointer_cast(u_wh.data());
    const double *um = thrust::raw_pointer_cast(u_mr.data());
    const double u_ref = dla::norm(2 * n_dof, uw);
    const double u_diff = dla::dist(2 * n_dof, uw, um);
    const double u_rel = (u_ref > 0) ? u_diff / u_ref : u_diff;

    const auto test_name = std::format("Postprocessing consistency ({})", precision);
    if (u_rel <= pass_rtol)
        summary.pass(test_name);
    else
        summary.fail(test_name, std::format("rel_diff = {:.2e}, tolerance = {:.2e}", u_rel, pass_rtol));
}

int main()
{
    TestLogger summary;
    test_operator_consistency<double>(summary, "double", 1e-4);
    test_operator_consistency<float>(summary, "float", 1e-2);

    test_rhs_evaluation<double>(summary, "double", 1e-6);
    test_rhs_evaluation<float>(summary, "float", 1e-4);

    test_postprocessing<double>(summary, "double", 1e-6);
    test_postprocessing<float>(summary, "float", 1e-4);

    return summary.finish();
}
