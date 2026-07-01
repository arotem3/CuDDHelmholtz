#include <thrust/copy.h>
#include <thrust/device_vector.h>

#include <numbers>

#include "test_common.hpp"

using namespace cuddh;

// Simple diagonal blocks: A_p[i,i] = (p+2) + (i+1)i, RHS = A_p[i,i], exact sol = 1+0i.
static void test_diagonal_batch(TestLogger &log)
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
    cudaDeviceSynchronize();

    std::vector<double> h_x(n_blocks * stride);
    thrust::copy(d_x.begin(), d_x.end(), h_x.begin());

    double err = 0.0;
    for (int p = 0; p < n_blocks; ++p)
        for (int i = 0; i < sizes[p]; ++i)
        {
            double dre = h_x[p * stride + i] - 1.0;
            double dim = h_x[p * stride + max_n + i];
            err = std::max(err, std::abs(dre));
            err = std::max(err, std::abs(dim));
        }

    if (err < 1e-10)
        log.pass("SparseBlockLU diagonal batch (3 blocks, exact sol = 1+0i)");
    else
        log.fail("SparseBlockLU diagonal batch (3 blocks, exact sol = 1+0i)", std::format("L∞ error = {:.2e}", err));
}

// Complex-shifted 2D FD diffusion in batch.
// Each block p solves (-Δ + iσ)u = f on a n_p×n_p interior grid (Dirichlet BCs).
// Exact solution u = sin(πx)sin(πy) [real], f = (2π² + iσ)u.
// The 5-point stencil (scaled by h²) has diagonal (4 + iσh²) and off-diagonals -1.
template <typename scalar_t>
static void test_fd_batch(TestLogger &log, std::string_view name)
{
    const int grid_ns[] = {5, 8, 10, 12, 15};
    const int n_blocks = 5;
    const scalar_t sigma = scalar_t(1);
    const scalar_t pi = std::numbers::pi_v<scalar_t>;
    using cx = std::complex<scalar_t>;

    int block_sizes[n_blocks];
    for (int p = 0; p < n_blocks; ++p)
        block_sizes[p] = grid_ns[p] * grid_ns[p];
    const int max_n = block_sizes[n_blocks - 1]; // 225 (15×15)
    const int stride = 2 * max_n;

    BlockSparseMatrix<scalar_t, true> bsm(n_blocks, block_sizes);

    for (int p = 0; p < n_blocks; ++p)
    {
        const int nn = grid_ns[p];
        for (int i = 0; i < nn; ++i)
            for (int j = 0; j < nn; ++j)
            {
                const int r = i * nn + j;
                bsm.add_entry(p, r, r);
                if (i > 0)
                    bsm.add_entry(p, r, r - nn);
                if (i < nn - 1)
                    bsm.add_entry(p, r, r + nn);
                if (j > 0)
                    bsm.add_entry(p, r, r - 1);
                if (j < nn - 1)
                    bsm.add_entry(p, r, r + 1);
            }
    }
    bsm.finalize_pattern();

    for (int p = 0; p < n_blocks; ++p)
    {
        const int nn = grid_ns[p];
        const scalar_t h = scalar_t(1) / (nn + 1);
        const scalar_t h2 = h * h;
        for (int i = 0; i < nn; ++i)
            for (int j = 0; j < nn; ++j)
            {
                const int r = i * nn + j;
                bsm.set_value(p, r, r, cx(scalar_t(4), sigma * h2));
                if (i > 0)
                    bsm.set_value(p, r, r - nn, cx(scalar_t(-1), scalar_t(0)));
                if (i < nn - 1)
                    bsm.set_value(p, r, r + nn, cx(scalar_t(-1), scalar_t(0)));
                if (j > 0)
                    bsm.set_value(p, r, r - 1, cx(scalar_t(-1), scalar_t(0)));
                if (j < nn - 1)
                    bsm.set_value(p, r, r + 1, cx(scalar_t(-1), scalar_t(0)));
            }
    }
    bsm.finalize_values();

    // Blocked RHS: [re_block0, ..., re_blockN, im_block0, ..., im_blockN]
    // For block p: f_r = h²*(2π² + iσ)*sin(πx_r)sin(πy_r)
    std::vector<scalar_t> h_rhs(n_blocks * stride, scalar_t(0));
    for (int p = 0; p < n_blocks; ++p)
    {
        const int nn = grid_ns[p];
        const scalar_t h = scalar_t(1) / (nn + 1);
        const scalar_t h2 = h * h;
        for (int i = 0; i < nn; ++i)
            for (int j = 0; j < nn; ++j)
            {
                const int r = i * nn + j;
                const scalar_t x = (i + 1) * h;
                const scalar_t y = (j + 1) * h;
                const scalar_t u = std::sin(pi * x) * std::sin(pi * y);
                h_rhs[p * stride + r] = h2 * scalar_t(2) * pi * pi * u;
                h_rhs[p * stride + max_n + r] = h2 * sigma * u;
            }
    }

    SparseBlockLU<scalar_t, true> blu(bsm);

    thrust::device_vector<scalar_t> d_rhs(h_rhs.begin(), h_rhs.end());
    thrust::device_vector<scalar_t> d_sol(n_blocks * stride, scalar_t(0));
    blu.solve(thrust::raw_pointer_cast(d_rhs.data()), thrust::raw_pointer_cast(d_sol.data()));
    cudaDeviceSynchronize();

    std::vector<scalar_t> h_sol(n_blocks * stride);
    thrust::copy(d_sol.begin(), d_sol.end(), h_sol.begin());

    // FD truncation error is O(h²); tightest case is n=5, h=1/6, error ≈ 2.2e-2.
    const scalar_t tol = scalar_t(5e-2);
    scalar_t re_err = scalar_t(0), im_err = scalar_t(0);
    for (int p = 0; p < n_blocks; ++p)
    {
        const int nn = grid_ns[p];
        const scalar_t h = scalar_t(1) / (nn + 1);
        for (int i = 0; i < nn; ++i)
            for (int j = 0; j < nn; ++j)
            {
                const int r = i * nn + j;
                const scalar_t x = (i + 1) * h;
                const scalar_t y = (j + 1) * h;
                const scalar_t u = std::sin(pi * x) * std::sin(pi * y);
                re_err = std::max(re_err, std::abs(h_sol[p * stride + r] - u));
                im_err = std::max(im_err, std::abs(h_sol[p * stride + max_n + r]));
            }
    }

    if (re_err < tol && im_err < tol)
        log.pass(std::format("SparseBlockLU complex-shifted FD batch 5×5..15×15 ({})", name));
    else
        log.fail(std::format("SparseBlockLU complex-shifted FD batch 5×5..15×15 ({})", name),
                 std::format("L∞ err re={:.3e} im={:.3e} (tol {:.3e})", double(re_err), double(im_err), double(tol)));
}

// Verify SparseBlockLU honours the SparseMatrixType passed to BlockSparseMatrix.
// The complex-shifted FD stencil satisfies A = A^T (complex symmetric) because:
//   diagonal: 4 + iσh² (same on both sides)
//   off-diagonals: -1 (symmetric by construction)
// CuDSS uses LDL^T when type = Symmetric, so the solve must still be correct.
static void test_batch_matrix_type(TestLogger &log)
{
    const int grid_ns[] = {5, 8, 10};
    const int n_blocks = 3;
    const double sigma = 1.0;
    const double pi = std::numbers::pi_v<double>;
    using cd = std::complex<double>;

    int block_sizes[n_blocks];
    for (int p = 0; p < n_blocks; ++p)
        block_sizes[p] = grid_ns[p] * grid_ns[p];
    const int max_n = block_sizes[n_blocks - 1];
    const int stride = 2 * max_n;

    BlockSparseMatrix<double, true> bsm(n_blocks, block_sizes, SparseMatrixType::Symmetric);

    for (int p = 0; p < n_blocks; ++p)
    {
        const int nn = grid_ns[p];
        for (int i = 0; i < nn; ++i)
            for (int j = 0; j < nn; ++j)
            {
                const int r = i * nn + j;
                bsm.add_entry(p, r, r);
                if (i > 0)
                    bsm.add_entry(p, r, r - nn);
                if (i < nn - 1)
                    bsm.add_entry(p, r, r + nn);
                if (j > 0)
                    bsm.add_entry(p, r, r - 1);
                if (j < nn - 1)
                    bsm.add_entry(p, r, r + 1);
            }
    }
    bsm.finalize_pattern();

    for (int p = 0; p < n_blocks; ++p)
    {
        const int nn = grid_ns[p];
        const double h = 1.0 / (nn + 1), h2 = h * h;
        for (int i = 0; i < nn; ++i)
            for (int j = 0; j < nn; ++j)
            {
                const int r = i * nn + j;
                bsm.set_value(p, r, r, cd(4.0, sigma * h2));
                if (i > 0)
                    bsm.set_value(p, r, r - nn, cd(-1.0, 0.0));
                if (i < nn - 1)
                    bsm.set_value(p, r, r + nn, cd(-1.0, 0.0));
                if (j > 0)
                    bsm.set_value(p, r, r - 1, cd(-1.0, 0.0));
                if (j < nn - 1)
                    bsm.set_value(p, r, r + 1, cd(-1.0, 0.0));
            }
    }
    bsm.finalize_values();

    std::vector<double> h_rhs(n_blocks * stride, 0.0);
    for (int p = 0; p < n_blocks; ++p)
    {
        const int nn = grid_ns[p];
        const double h = 1.0 / (nn + 1), h2 = h * h;
        for (int i = 0; i < nn; ++i)
            for (int j = 0; j < nn; ++j)
            {
                const int r = i * nn + j;
                const double x = (i + 1) * h, y = (j + 1) * h;
                const double u = std::sin(pi * x) * std::sin(pi * y);
                h_rhs[p * stride + r] = h2 * 2.0 * pi * pi * u;
                h_rhs[p * stride + max_n + r] = h2 * sigma * u;
            }
    }

    SparseBlockLU<double, true> blu(bsm);

    thrust::device_vector<double> d_rhs(h_rhs.begin(), h_rhs.end());
    thrust::device_vector<double> d_sol(n_blocks * stride, 0.0);
    blu.solve(thrust::raw_pointer_cast(d_rhs.data()), thrust::raw_pointer_cast(d_sol.data()));
    cudaDeviceSynchronize();

    std::vector<double> h_sol(n_blocks * stride);
    thrust::copy(d_sol.begin(), d_sol.end(), h_sol.begin());

    constexpr double tol = 5e-2;
    double re_err = 0.0, im_err = 0.0;
    for (int p = 0; p < n_blocks; ++p)
    {
        const int nn = grid_ns[p];
        const double h = 1.0 / (nn + 1);
        for (int i = 0; i < nn; ++i)
            for (int j = 0; j < nn; ++j)
            {
                const int r = i * nn + j;
                const double x = (i + 1) * h, y = (j + 1) * h;
                const double u = std::sin(pi * x) * std::sin(pi * y);
                re_err = std::max(re_err, std::abs(h_sol[p * stride + r] - u));
                im_err = std::max(im_err, std::abs(h_sol[p * stride + max_n + r]));
            }
    }

    if (re_err < tol && im_err < tol)
        log.pass("SparseBlockLU matrix type: Symmetric (complex-shifted FD, LDL^T)");
    else
        log.fail("SparseBlockLU matrix type: Symmetric (complex-shifted FD, LDL^T)",
                 std::format("L∞ err re={:.3e} im={:.3e} (tol {:.3e})", re_err, im_err, tol));
}

int main()
{
    TestLogger log;
    test_diagonal_batch(log);
    test_fd_batch<double>(log, "double");
    test_fd_batch<float>(log, "float");
    test_batch_matrix_type(log);
    return log.finish();
}
