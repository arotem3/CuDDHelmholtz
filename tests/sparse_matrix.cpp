#include <thrust/copy.h>
#include <thrust/device_vector.h>

#include <numbers>

#include "test_common.hpp"

using namespace cuddh;

template <typename scalar_t>
static void test_basic_assembly_and_action(TestLogger &summary, std::string_view name)
{
    SparseMatrix<scalar_t, false> A(3, 3, 5);
    A.add_entry(0, 0);
    A.add_entry(0, 1);
    A.add_entry(1, 1);
    A.add_entry(1, 2);
    A.add_entry(2, 2);

    A.finalize_pattern();
    A.set_value(0, 0, scalar_t(2));
    A.set_value(0, 1, scalar_t(-1));
    A.set_value(1, 1, scalar_t(3));
    A.set_value(1, 2, scalar_t(4));
    A.set_value(2, 2, scalar_t(5));

    A.finalize_values();

    const scalar_t h_x[3] = {scalar_t(1), scalar_t(2), scalar_t(3)};
    thrust::device_vector<scalar_t> d_x(h_x, h_x + 3);
    thrust::device_vector<scalar_t> d_y(3, scalar_t(0));

    A.action(thrust::raw_pointer_cast(d_x.data()), thrust::raw_pointer_cast(d_y.data()));
    cudaDeviceSynchronize();

    std::vector<scalar_t> y(3);
    thrust::copy(d_y.begin(), d_y.end(), y.begin());

    const scalar_t ref0 = scalar_t(0);  // 2*1 - 1*2
    const scalar_t ref1 = scalar_t(18); // 3*2 + 4*3
    const scalar_t ref2 = scalar_t(15); // 5*3

    const bool ok = (std::abs(y[0] - ref0) < scalar_t(1e-10)) && (std::abs(y[1] - ref1) < scalar_t(1e-10)) &&
                    (std::abs(y[2] - ref2) < scalar_t(1e-10));

    if (ok)
        summary.pass(std::format("Sparse action in finalized state ({})", name));
    else
        summary.fail(std::format("Sparse action in finalized state ({})", name),
                     std::format("got [{}, {}, {}]", y[0], y[1], y[2]));

    thrust::fill(d_y.begin(), d_y.end(), scalar_t(0));
    A.action(thrust::raw_pointer_cast(d_x.data()), thrust::raw_pointer_cast(d_y.data()));
    cudaDeviceSynchronize();
    thrust::copy(d_y.begin(), d_y.end(), y.begin());

    const bool ok2 = (std::abs(y[0] - ref0) < scalar_t(1e-10)) && (std::abs(y[1] - ref1) < scalar_t(1e-10)) &&
                     (std::abs(y[2] - ref2) < scalar_t(1e-10));

    if (ok2)
        summary.pass(std::format("Sparse action in finalized state ({})", name));
    else
        summary.fail(std::format("Sparse action in finalized state ({})", name),
                     std::format("got [{}, {}, {}]", y[0], y[1], y[2]));
}

static void test_state_transitions(TestLogger &summary)
{
    bool ok = true;

    SparseLU<double> lu = [&ok]() {
        SparseMatrix<double, false> A(2, 2, 2);
        A.add_entry(0, 0);
        A.add_entry(1, 1);

        ok = ok && (A.state() == SparseMatrixState::PatternAssembly);

        A.finalize_pattern();
        ok = ok && (A.state() == SparseMatrixState::COOAssembly);
        A.set_value(0, 0, 1.0);
        A.set_value(1, 1, 2.0);

        A.finalize_values();
        ok = ok && (A.state() == SparseMatrixState::Finalized);

        return SparseLU<double, false>(A);
    }();

    const auto &stats = lu.stats();
    ok = ok && (stats.solve_calls == 0);

    const double h_rhs[2] = {1.0, 2.0};
    thrust::device_vector<double> d_rhs(h_rhs, h_rhs + 2);
    thrust::device_vector<double> d_sol(2, 0.0);

    ok = ok && lu.solve(thrust::raw_pointer_cast(d_rhs.data()), thrust::raw_pointer_cast(d_sol.data()));
    cudaDeviceSynchronize();
    std::vector<double> sol(2);
    thrust::copy(d_sol.begin(), d_sol.end(), sol.begin());
    ok = ok && (std::abs(sol[0] - 1.0) < 1e-12) && (std::abs(sol[1] - 1.0) < 1e-12);
    ok = ok && (lu.stats().solve_calls == 1);

    thrust::fill(d_sol.begin(), d_sol.end(), 0.0);
    ok = ok && lu.solve(thrust::raw_pointer_cast(d_rhs.data()), thrust::raw_pointer_cast(d_sol.data()));
    cudaDeviceSynchronize();
    thrust::copy(d_sol.begin(), d_sol.end(), sol.begin());
    ok = ok && (std::abs(sol[0] - 1.0) < 1e-12) && (std::abs(sol[1] - 1.0) < 1e-12);
    ok = ok && (lu.stats().solve_calls == 2);

    if (ok)
        summary.pass("Sparse matrix state transitions and SparseLU matrix-lifetime independence");
    else
        summary.fail("Sparse matrix state transitions and SparseLU matrix-lifetime independence",
                     "unexpected state transition result");
}

static void test_complex_assembly(TestLogger &summary)
{
    using cdouble = std::complex<double>;

    SparseMatrix<double, true> A(2, 2, 2);
    A.add_entry(0, 0);
    A.add_entry(1, 1);

    A.finalize_pattern();
    A.set_value(0, 0, cdouble(1.0, 2.0));
    A.set_value(1, 1, cdouble(3.0, -1.0));

    A.finalize_values();

    // x in blocked format: [re(1+0i), re(0+1i), im(1+0i), im(0+1i)] = [1, 0, 0, 1]
    const double h_x[4] = {1.0, 0.0, 0.0, 1.0};
    thrust::device_vector<double> d_x(h_x, h_x + 4);
    thrust::device_vector<double> d_y(4, 0.0);
    A.action(thrust::raw_pointer_cast(d_x.data()), thrust::raw_pointer_cast(d_y.data()));
    cudaDeviceSynchronize();
    std::vector<double> y(4);
    thrust::copy(d_y.begin(), d_y.end(), y.begin());

    const cdouble y0(y[0], y[2]);
    const cdouble y1(y[1], y[3]);

    const cdouble r0 = cdouble(1.0, 2.0);
    const cdouble r1 = cdouble(1.0, 3.0);

    const bool ok = (std::abs(y0 - r0) < 1e-12) && (std::abs(y1 - r1) < 1e-12);
    const std::string got = std::string("got [") + std::to_string(y0.real()) + ", " + std::to_string(y0.imag()) + "; " +
                            std::to_string(y1.real()) + ", " + std::to_string(y1.imag()) + "]";

    if (ok)
        summary.pass("Sparse complex assembly and action");
    else
        summary.fail("Sparse complex assembly and action", got);
}

static void test_factor_and_solve(TestLogger &summary)
{
    SparseMatrix<double, false> A(2, 2, 2);
    A.add_entry(0, 0);
    A.add_entry(1, 1);
    A.finalize_pattern();
    A.set_value(0, 0, 2.0);
    A.set_value(1, 1, 3.0);
    A.finalize_values();

    SparseLU<double, false> lu(A);

    bool ok = (lu.stats().analysis_seconds >= 0.0) && (lu.stats().factor_seconds >= 0.0);
    ok = ok && (lu.stats().finalized_bytes > 0);
    ok = ok && (lu.stats().factor_bytes > 0);

    {
        const double h_rhs[2] = {2.0, 6.0};
        thrust::device_vector<double> d_rhs(h_rhs, h_rhs + 2);
        thrust::device_vector<double> d_sol(2, 0.0);
        ok = ok && lu.solve(thrust::raw_pointer_cast(d_rhs.data()), thrust::raw_pointer_cast(d_sol.data()));
        cudaDeviceSynchronize();
        std::vector<double> sol(2);
        thrust::copy(d_sol.begin(), d_sol.end(), sol.begin());
        ok = ok && (std::abs(sol[0] - 1.0) < 1e-12) && (std::abs(sol[1] - 2.0) < 1e-12);
        ok = ok && (lu.stats().solve_calls == 1);
        ok = ok && (lu.stats().total_solve_seconds >= 0.0);

        SparseMatrix<double, true> C(1, 1, 1);
        C.add_entry(0, 0);
        C.finalize_pattern();
        C.set_value(0, 0, std::complex<double>(2.0, 1.0));
        C.finalize_values();
        SparseLU<double, true> clu(C);
        // RHS = (2+1i) in blocked format: [re=2, im=1]. Solution = 1+0i.
        const double h_crhs[2] = {2.0, 1.0};
        thrust::device_vector<double> d_crhs(h_crhs, h_crhs + 2);
        thrust::device_vector<double> d_csol(2, 0.0);
        ok = ok && clu.solve(thrust::raw_pointer_cast(d_crhs.data()), thrust::raw_pointer_cast(d_csol.data()));
        cudaDeviceSynchronize();
        std::vector<double> csol(2);
        thrust::copy(d_csol.begin(), d_csol.end(), csol.begin());
        ok = ok && (std::abs(csol[0] - 1.0) < 1e-12) && (std::abs(csol[1] - 0.0) < 1e-12);
        ok = ok && (clu.stats().solve_calls == 1);

        if (ok)
            summary.pass("SparseLU factor, solve, and stats (real + blocked-complex)");
        else
            summary.fail("SparseLU factor, solve, and stats (real + blocked-complex)",
                         std::format("got real [{:.3e}, {:.3e}] and complex [{:.3e}, {:.3e}]", sol[0], sol[1], csol[0],
                                     csol[1]));
    }
}

// Verify SparseBlockLU batch solve: non-uniform diagonal blocks with analytic solution = 1+0i.
// Block p: size n_p, diagonal A_p[i,i] = (p+2) + (i+1)*j, RHS = A_p[i,i], expected x = 1+0i.
static void test_block_factor_and_solve(TestLogger &log)
{
    using cdouble = std::complex<double>;

    const int sizes[] = {2, 3, 2};
    const int n_blocks = 3;
    const int max_n = 3;
    const int stride = 2 * max_n;

    BlockSparseMatrix<double, true> bsm(n_blocks, sizes);
    for (int p = 0; p < n_blocks; ++p)
        for (int i = 0; i < sizes[p]; ++i)
            bsm.add_entry(p, i, i);
    bsm.finalize_pattern();
    for (int p = 0; p < n_blocks; ++p)
        for (int i = 0; i < sizes[p]; ++i)
            bsm.set_value(p, i, i, cdouble(p + 2.0, i + 1.0));
    bsm.finalize_values();

    // RHS in blocked format: rhs_p[i] = A_p[i,i]; analytic solution is all 1+0i.
    std::vector<double> h_rhs(n_blocks * stride, 0.0);
    for (int p = 0; p < n_blocks; ++p)
        for (int i = 0; i < sizes[p]; ++i)
        {
            h_rhs[p * stride + i] = p + 2.0;
            h_rhs[p * stride + max_n + i] = i + 1.0;
        }

    SparseBlockLU<double, true> blu(bsm);

    thrust::device_vector<double> d_rhs(h_rhs.begin(), h_rhs.end());
    thrust::device_vector<double> d_x(n_blocks * stride, 0.0);
    blu.solve(thrust::raw_pointer_cast(d_rhs.data()), thrust::raw_pointer_cast(d_x.data()));

    std::vector<double> h_x(n_blocks * stride);
    thrust::copy(d_x.begin(), d_x.end(), h_x.begin());

    double err = 0.0;
    for (int p = 0; p < n_blocks; ++p)
        for (int i = 0; i < sizes[p]; ++i)
        {
            double dre = h_x[p * stride + i] - 1.0;
            double dim = h_x[p * stride + max_n + i] - 0.0;
            err += dre * dre + dim * dim;
        }
    const double rmse = std::sqrt(err);

    if (rmse < 1e-10)
        log.pass("SparseBlockLU batch solve (analytic check, all solution entries = 1+0i)");
    else
    {
        char buf[64];
        std::snprintf(buf, sizeof(buf), "RMSE %.2e", rmse);
        log.fail("SparseBlockLU batch solve (analytic check, all solution entries = 1+0i)", buf);
    }
}

// ─── SparseMatrixType tests ──────────────────────────────────────────────────
//
// CuDSS selects a different factorization algorithm for each matrix type:
//   General   → LU
//   Symmetric → LDL^T (real or complex symmetric)
//   SPD       → Cholesky (real)
//   Hermitian → LDL^H
//   HPD       → Cholesky^H (complex Hermitian positive definite)
//
// Each test below uses data that genuinely satisfies the declared type.

// The 10×10 FD Laplacian (-Δ_h) is real symmetric and SPD — valid for Symmetric and SPD.
// Returns the L∞ error vs sin(πx)sin(πy).
static double solve_fd_real(SparseMatrixType type)
{
    const int n = 10, N = n * n;
    const double h = 1.0 / (n + 1), h2 = h * h;
    const double pi = std::numbers::pi_v<double>;

    SparseMatrix<double, false> A(N, N, 5 * N, type);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
        {
            const int r = i * n + j;
            A.add_entry(r, r);
            if (i > 0)
                A.add_entry(r, r - n);
            if (i < n - 1)
                A.add_entry(r, r + n);
            if (j > 0)
                A.add_entry(r, r - 1);
            if (j < n - 1)
                A.add_entry(r, r + 1);
        }
    A.finalize_pattern();
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
        {
            const int r = i * n + j;
            A.set_value(r, r, 4.0);
            if (i > 0)
                A.set_value(r, r - n, -1.0);
            if (i < n - 1)
                A.set_value(r, r + n, -1.0);
            if (j > 0)
                A.set_value(r, r - 1, -1.0);
            if (j < n - 1)
                A.set_value(r, r + 1, -1.0);
        }
    A.finalize_values();

    std::vector<double> h_rhs(N), h_exact(N);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
        {
            const double x = (i + 1) * h, y = (j + 1) * h;
            const double u = std::sin(pi * x) * std::sin(pi * y);
            h_exact[i * n + j] = u;
            h_rhs[i * n + j] = h2 * 2.0 * pi * pi * u;
        }

    SparseLU<double, false> lu(A);
    thrust::device_vector<double> d_rhs(h_rhs.begin(), h_rhs.end());
    thrust::device_vector<double> d_sol(N, 0.0);
    lu.solve(thrust::raw_pointer_cast(d_rhs.data()), thrust::raw_pointer_cast(d_sol.data()));
    cudaDeviceSynchronize();

    std::vector<double> h_sol(N);
    thrust::copy(d_sol.begin(), d_sol.end(), h_sol.begin());

    double err = 0.0;
    for (int k = 0; k < N; ++k)
        err = std::max(err, std::abs(h_sol[k] - h_exact[k]));
    return err;
}

// Complex sparse matrix holding the FD Laplacian with an optional imaginary shift σ.
//   σ = 0 → matrix is purely real → Hermitian (A^H = A^T = A) and HPD
//   σ > 0 → complex-shifted → complex symmetric (A = A^T, but A ≠ A^H)
// Returns {re_err, im_err} vs (sin(πx)sin(πy), 0).
static std::pair<double, double> solve_fd_cx(SparseMatrixType type, double sigma)
{
    const int n = 10, N = n * n;
    const double h = 1.0 / (n + 1), h2 = h * h;
    const double pi = std::numbers::pi_v<double>;
    using cd = std::complex<double>;

    SparseMatrix<double, true> A(N, N, 5 * N, type);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
        {
            const int r = i * n + j;
            A.add_entry(r, r);
            if (i > 0)
                A.add_entry(r, r - n);
            if (i < n - 1)
                A.add_entry(r, r + n);
            if (j > 0)
                A.add_entry(r, r - 1);
            if (j < n - 1)
                A.add_entry(r, r + 1);
        }
    A.finalize_pattern();
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
        {
            const int r = i * n + j;
            A.set_value(r, r, cd(4.0, sigma * h2));
            if (i > 0)
                A.set_value(r, r - n, cd(-1.0, 0.0));
            if (i < n - 1)
                A.set_value(r, r + n, cd(-1.0, 0.0));
            if (j > 0)
                A.set_value(r, r - 1, cd(-1.0, 0.0));
            if (j < n - 1)
                A.set_value(r, r + 1, cd(-1.0, 0.0));
        }
    A.finalize_values();

    // f = h²*(2π² + iσ)*u_exact, RHS in blocked format
    std::vector<double> h_rhs(2 * N, 0.0), h_exact(N);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
        {
            const int k = i * n + j;
            const double x = (i + 1) * h, y = (j + 1) * h;
            const double u = std::sin(pi * x) * std::sin(pi * y);
            h_exact[k] = u;
            h_rhs[k] = h2 * 2.0 * pi * pi * u;
            h_rhs[N + k] = h2 * sigma * u;
        }

    SparseLU<double, true> lu(A);
    thrust::device_vector<double> d_rhs(h_rhs.begin(), h_rhs.end());
    thrust::device_vector<double> d_sol(2 * N, 0.0);
    lu.solve(thrust::raw_pointer_cast(d_rhs.data()), thrust::raw_pointer_cast(d_sol.data()));
    cudaDeviceSynchronize();

    std::vector<double> h_sol(2 * N);
    thrust::copy(d_sol.begin(), d_sol.end(), h_sol.begin());

    double re_err = 0.0, im_err = 0.0;
    for (int k = 0; k < N; ++k)
    {
        re_err = std::max(re_err, std::abs(h_sol[k] - h_exact[k]));
        im_err = std::max(im_err, std::abs(h_sol[N + k]));
    }
    return {re_err, im_err};
}

static void test_matrix_types(TestLogger &log)
{
    constexpr double tol = 5e-2;

    auto check_real = [&](SparseMatrixType type, std::string_view label) {
        const double err = solve_fd_real(type);
        if (err < tol)
            log.pass(std::format("SparseLU matrix type: {}", label));
        else
            log.fail(std::format("SparseLU matrix type: {}", label),
                     std::format("L∞ error = {:.3e} (tol {:.3e})", err, tol));
    };

    auto check_cx = [&](SparseMatrixType type, double sigma, std::string_view label) {
        auto [re_err, im_err] = solve_fd_cx(type, sigma);
        if (re_err < tol && im_err < tol)
            log.pass(std::format("SparseLU matrix type: {}", label));
        else
            log.fail(std::format("SparseLU matrix type: {}", label),
                     std::format("L∞ err re={:.3e} im={:.3e} (tol {:.3e})", re_err, im_err, tol));
    };

    // Real matrix types (SparseMatrix<double, false>)
    check_real(SparseMatrixType::Symmetric, "Symmetric (real FD, A=A^T)");
    check_real(SparseMatrixType::SPD, "SPD       (real FD, Cholesky)");

    // Complex matrix types (SparseMatrix<double, true>)
    // σ=1: complex-shifted FD is complex symmetric (A=A^T, not A^H)
    check_cx(SparseMatrixType::Symmetric, 1.0, "Symmetric (complex-shifted FD, A=A^T)");
    // σ=0: purely real FD embedded as complex; A^H = A^T = A → Hermitian and HPD
    check_cx(SparseMatrixType::Hermitian, 0.0, "Hermitian (real FD as complex, A^H=A)");
    check_cx(SparseMatrixType::HPD, 0.0, "HPD       (real FD as complex, Cholesky^H)");
}

// 2D FD diffusion -Δu = f on [0,1]² with Dirichlet BCs.
// Exact solution u = sin(πx)sin(πy), f = 2π²u.
// The 5-point stencil (scaled by h²) has diagonal 4 and off-diagonals -1.
template <typename scalar_t>
static void test_fd_real_diffusion(TestLogger &log, std::string_view name)
{
    const int n = 10;
    const int N = n * n;
    const scalar_t h = scalar_t(1) / (n + 1);
    const scalar_t h2 = h * h;
    const scalar_t pi = std::numbers::pi_v<scalar_t>;

    SparseMatrix<scalar_t, false> A(N, N, 5 * N);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
        {
            const int r = i * n + j;
            A.add_entry(r, r);
            if (i > 0)
                A.add_entry(r, r - n);
            if (i < n - 1)
                A.add_entry(r, r + n);
            if (j > 0)
                A.add_entry(r, r - 1);
            if (j < n - 1)
                A.add_entry(r, r + 1);
        }
    A.finalize_pattern();
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
        {
            const int r = i * n + j;
            A.set_value(r, r, scalar_t(4));
            if (i > 0)
                A.set_value(r, r - n, scalar_t(-1));
            if (i < n - 1)
                A.set_value(r, r + n, scalar_t(-1));
            if (j > 0)
                A.set_value(r, r - 1, scalar_t(-1));
            if (j < n - 1)
                A.set_value(r, r + 1, scalar_t(-1));
        }
    A.finalize_values();

    std::vector<scalar_t> h_rhs(N), h_exact(N);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
        {
            const scalar_t x = (i + 1) * h;
            const scalar_t y = (j + 1) * h;
            const scalar_t u = std::sin(pi * x) * std::sin(pi * y);
            h_exact[i * n + j] = u;
            h_rhs[i * n + j] = h2 * scalar_t(2) * pi * pi * u;
        }

    SparseLU<scalar_t, false> lu(A);

    thrust::device_vector<scalar_t> d_rhs(h_rhs.begin(), h_rhs.end());
    thrust::device_vector<scalar_t> d_sol(N, scalar_t(0));
    const bool solved = lu.solve(thrust::raw_pointer_cast(d_rhs.data()), thrust::raw_pointer_cast(d_sol.data()));
    cudaDeviceSynchronize();

    std::vector<scalar_t> h_sol(N);
    thrust::copy(d_sol.begin(), d_sol.end(), h_sol.begin());

    scalar_t max_err = scalar_t(0);
    for (int k = 0; k < N; ++k)
        max_err = std::max(max_err, std::abs(h_sol[k] - h_exact[k]));

    // FD truncation error is O(h²) ≈ 7e-3 for n=10; use 5e-2 as safe bound.
    const scalar_t tol = scalar_t(5e-2);
    if (solved && max_err < tol)
        log.pass(std::format("SparseLU real FD diffusion 10×10 ({})", name));
    else
        log.fail(std::format("SparseLU real FD diffusion 10×10 ({})", name),
                 std::format("L∞ error = {:.3e} (tol {:.3e})", double(max_err), double(tol)));
}

// Complex-shifted FD: (-Δ + iσ)u = f, u = sin(πx)sin(πy) [real].
// The shift makes the diagonal complex: 4 + iσh².
template <typename scalar_t>
static void test_fd_cx_diffusion(TestLogger &log, std::string_view name)
{
    const int n = 10;
    const int N = n * n;
    const scalar_t h = scalar_t(1) / (n + 1);
    const scalar_t h2 = h * h;
    const scalar_t pi = std::numbers::pi_v<scalar_t>;
    const scalar_t sigma = scalar_t(1);
    using cx = std::complex<scalar_t>;

    SparseMatrix<scalar_t, true> A(N, N, 5 * N);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
        {
            const int r = i * n + j;
            A.add_entry(r, r);
            if (i > 0)
                A.add_entry(r, r - n);
            if (i < n - 1)
                A.add_entry(r, r + n);
            if (j > 0)
                A.add_entry(r, r - 1);
            if (j < n - 1)
                A.add_entry(r, r + 1);
        }
    A.finalize_pattern();
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
        {
            const int r = i * n + j;
            A.set_value(r, r, cx(scalar_t(4), sigma * h2));
            if (i > 0)
                A.set_value(r, r - n, cx(scalar_t(-1), scalar_t(0)));
            if (i < n - 1)
                A.set_value(r, r + n, cx(scalar_t(-1), scalar_t(0)));
            if (j > 0)
                A.set_value(r, r - 1, cx(scalar_t(-1), scalar_t(0)));
            if (j < n - 1)
                A.set_value(r, r + 1, cx(scalar_t(-1), scalar_t(0)));
        }
    A.finalize_values();

    // RHS in blocked format: f = h²*(2π² + iσ)*u_exact
    std::vector<scalar_t> h_rhs(2 * N), h_exact(N);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
        {
            const int k = i * n + j;
            const scalar_t x = (i + 1) * h;
            const scalar_t y = (j + 1) * h;
            const scalar_t u = std::sin(pi * x) * std::sin(pi * y);
            h_exact[k] = u;
            h_rhs[k] = h2 * scalar_t(2) * pi * pi * u; // Re(f)
            h_rhs[N + k] = h2 * sigma * u;             // Im(f)
        }

    SparseLU<scalar_t, true> lu(A);

    thrust::device_vector<scalar_t> d_rhs(h_rhs.begin(), h_rhs.end());
    thrust::device_vector<scalar_t> d_sol(2 * N, scalar_t(0));
    const bool solved = lu.solve(thrust::raw_pointer_cast(d_rhs.data()), thrust::raw_pointer_cast(d_sol.data()));
    cudaDeviceSynchronize();

    std::vector<scalar_t> h_sol(2 * N);
    thrust::copy(d_sol.begin(), d_sol.end(), h_sol.begin());

    scalar_t re_err = scalar_t(0), im_err = scalar_t(0);
    for (int k = 0; k < N; ++k)
    {
        re_err = std::max(re_err, std::abs(h_sol[k] - h_exact[k]));
        im_err = std::max(im_err, std::abs(h_sol[N + k]));
    }

    const scalar_t tol = scalar_t(5e-2);
    if (solved && re_err < tol && im_err < tol)
        log.pass(std::format("SparseLU complex-shifted FD diffusion 10×10 ({})", name));
    else
        log.fail(std::format("SparseLU complex-shifted FD diffusion 10×10 ({})", name),
                 std::format("L∞ err re={:.3e} im={:.3e} (tol {:.3e})", double(re_err), double(im_err), double(tol)));
}

int main()
{
    TestLogger summary;

    test_basic_assembly_and_action<double>(summary, "double");
    test_complex_assembly(summary);
    test_state_transitions(summary);
    test_factor_and_solve(summary);
    test_block_factor_and_solve(summary);
    test_fd_real_diffusion<double>(summary, "double");
    test_fd_real_diffusion<float>(summary, "float");
    test_fd_cx_diffusion<double>(summary, "double");
    test_fd_cx_diffusion<float>(summary, "float");
    test_matrix_types(summary);

    return summary.finish();
}
