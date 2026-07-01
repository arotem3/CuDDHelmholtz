/**
 * @file dd_block_assembly.cpp
 * @brief Tests for per-block Helmholtz sparse matrix assembly from EnsembleSpace.
 *
 * Correctness strategy: a single-subdomain EnsembleSpace (all elements in subdomain 0)
 * must produce a per-block matrix A_0 that equals the global Helmholtz matrix A, up to
 * the local↔global DOF reordering given by EnsembleSpace::global_indices().
 *
 * The global matrix is assembled via Helmholtz::assemble (already verified in assemble.cpp).
 * For a random blocked-complex vector x, both paths must give the same A*x after reindexing.
 */

#include <complex>
#include <random>
#include <vector>

#include "test_common.hpp"

using namespace cuddh;

// CPU sparse matrix-vector product for one block of a BlockSparseMatrix<double,true>.
// Input x is in blocked format [x_re(0..n-1), x_im(0..n-1)]; same for output y.
static std::vector<double> block_matvec(const BlockSparseMatrix<double, true> &B, int p, const std::vector<double> &x)
{
    const int n = B.block_size(p);
    std::vector<double> y(2 * n, 0.0);
    const int *rp = B.row_ptrs() + B.rp_offset(p);
    const int *ci = B.col_indices() + B.nz_offset(p);
    const std::complex<double> *val = B.values() + B.nz_offset(p);
    for (int row = 0; row < n; ++row)
    {
        for (int k = rp[row]; k < rp[row + 1]; ++k)
        {
            const int col = ci[k];
            const double xr = x[col];
            const double xi = x[n + col];
            y[row] += val[k].real() * xr - val[k].imag() * xi;
            y[n + row] += val[k].real() * xi + val[k].imag() * xr;
        }
    }
    return y;
}

// Test that the per-block Helmholtz assembly matches the global Helmholtz assembly.
// Uses a single-subdomain partition so the local and global problems are identical
// (up to DOF reordering given by EnsembleSpace::global_indices).
static void test_single_subdomain_assembly(TestLogger &log)
{
    constexpr int N = 6;
    constexpr int DEG = 3;
    const double omega = 2.0;

    Mesh2D mesh = Mesh2D::uniform_rect(N, -1.0, 1.0, N, -1.0, 1.0);
    Basis basis(DEG + 1);
    H1Space2D fem(mesh, basis);
    const int ndof = fem.size();
    const int N_cx = 2 * ndof;

    // Single-subdomain partition: every element → subdomain 0
    std::vector<int> labels(mesh.n_elem(), 0);
    EnsembleSpace efem(fem, 1, labels.data());

    // Global Helmholtz with constant coefficient a(x) = 1
    ivec bfaces = mesh.boundary_edges();
    TraceSpace2D tr(fem, bfaces.size(), bfaces.data());
    Helmholtz A_global(fem, tr, omega);

    // Assemble global SparseMatrix<double,true>
    SparseMatrix<double, true> S_global(ndof, ndof);
    fem.set_pattern(S_global);
    S_global.finalize_pattern();
    A_global.assemble({1.0, 0.0}, S_global);
    S_global.finalize_values();

    // Assemble per-block BlockSparseMatrix (one block = whole domain)
    const auto blk_sizes = efem.sizes(MemorySpace::HOST);
    BlockSparseMatrix<double, true> blocks(1, blk_sizes.data());

    DDStiffnessMatrix<double> S_dd(efem);
    efem.set_pattern(blocks);
    blocks.finalize_pattern();

    DDMassMatrix<double> M_dd(efem);
    DDFaceMassMatrix<double> H_dd(efem);
    S_dd.assemble({1.0, 0.0}, blocks);
    M_dd.assemble({-omega * omega, 0.0}, blocks);
    H_dd.assemble({0.0, -omega}, blocks);
    blocks.finalize_values();

    // DOF mapping: local subdomain DOF k → global FEM DOF gI(k)
    const auto gI = efem.global_indices(MemorySpace::HOST);
    const int n_local = efem.sizes(MemorySpace::HOST)(0);

    // Random input in local DOF space [re; im]
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::vector<double> x_local(2 * n_local);
    for (auto &v : x_local)
        v = dist(rng);

    // Scatter x_local to global DOF space
    std::vector<double> x_global(N_cx, 0.0);
    for (int k = 0; k < n_local; ++k)
    {
        const int g = gI(k, 0);
        x_global[g] = x_local[k];
        x_global[ndof + g] = x_local[n_local + k];
    }

    // Apply per-block matrix (CPU matvec) → y_block in local DOF space
    const std::vector<double> y_block = block_matvec(blocks, 0, x_local);

    // Apply global sparse matrix (GPU) → y_global in global DOF space
    thrust::universal_vector<double> d_xg(x_global.begin(), x_global.end());
    thrust::universal_vector<double> d_yg(N_cx, 0.0);
    S_global.action(d_xg.data().get(), d_yg.data().get());
    cudaDeviceSynchronize();

    // Gather y_global to local DOF space via gI
    std::vector<double> y_ref(2 * n_local);
    for (int k = 0; k < n_local; ++k)
    {
        const int g = gI(k, 0);
        y_ref[k] = d_yg[g];
        y_ref[n_local + k] = d_yg[ndof + g];
    }

    // Compare y_block vs y_ref
    double err = 0.0, ref_norm = 0.0;
    for (int i = 0; i < 2 * n_local; ++i)
    {
        err = std::max(err, std::abs(y_block[i] - y_ref[i]));
        ref_norm = std::max(ref_norm, std::abs(y_ref[i]));
    }
    const double rel = (ref_norm > 0.0) ? err / ref_norm : err;

    if (rel < 1e-10)
        log.pass("BlockSparseMatrix single-subdomain assembly matches global Helmholtz (constant a)");
    else
        log.fail("BlockSparseMatrix single-subdomain assembly matches global Helmholtz (constant a)",
                 std::format("L∞ rel err = {:.2e}", rel));
}

// Same as above but with a variable coefficient a(x)
static void test_single_subdomain_assembly_variable_a(TestLogger &log)
{
    constexpr int N = 6;
    constexpr int DEG = 2;
    const double omega = 3.0;

    Mesh2D mesh = Mesh2D::uniform_rect(N, -1.0, 1.0, N, -1.0, 1.0);
    Basis basis(DEG + 1);
    H1Space2D fem(mesh, basis);
    const int ndof = fem.size();
    const int N_cx = 2 * ndof;

    std::vector<int> labels(mesh.n_elem(), 0);
    EnsembleSpace efem(fem, 1, labels.data());

    // Variable coefficient: a(x,y) = 1 + 0.5*x
    GridFunc2D<double> a_coeff(fem);
    {
        auto h_a = a_coeff.write(MemorySpace::HOST);
        // GridFunc2D stores values at quadrature nodes per element: h_a(q1, q2, el)
        // For simplicity fill with a constant slightly above 1 to exercise non-trivial a
        std::fill(h_a.begin(), h_a.end(), 1.5);
    }

    ivec bfaces = mesh.boundary_edges();
    TraceSpace2D tr(fem, bfaces.size(), bfaces.data());
    Helmholtz A_global(fem, tr, omega, a_coeff);

    SparseMatrix<double, true> S_global(ndof, ndof);
    fem.set_pattern(S_global);
    S_global.finalize_pattern();
    A_global.assemble({1.0, 0.0}, S_global);
    S_global.finalize_values();

    const auto blk_sizes = efem.sizes(MemorySpace::HOST);
    BlockSparseMatrix<double, true> blocks(1, blk_sizes.data());

    DDStiffnessMatrix<double> S_dd(efem);
    efem.set_pattern(blocks);
    blocks.finalize_pattern();

    // DDMassMatrix needs a^2 (same convention as Helmholtz, which uses square(a) for M)
    GridFunc2D<double> a_sq = a_coeff.transform([] __device__(double x) -> double { return x * x; });
    DDMassMatrix<double> M_dd(efem, a_sq);
    DDFaceMassMatrix<double> H_dd(efem, a_coeff);
    S_dd.assemble({1.0, 0.0}, blocks);
    M_dd.assemble({-omega * omega, 0.0}, blocks);
    H_dd.assemble({0.0, -omega}, blocks);
    blocks.finalize_values();

    const auto gI = efem.global_indices(MemorySpace::HOST);
    const int n_local = efem.sizes(MemorySpace::HOST)(0);

    std::mt19937 rng(99);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::vector<double> x_local(2 * n_local);
    for (auto &v : x_local)
        v = dist(rng);

    std::vector<double> x_global(N_cx, 0.0);
    for (int k = 0; k < n_local; ++k)
    {
        const int g = gI(k, 0);
        x_global[g] = x_local[k];
        x_global[ndof + g] = x_local[n_local + k];
    }

    const std::vector<double> y_block = block_matvec(blocks, 0, x_local);

    thrust::universal_vector<double> d_xg(x_global.begin(), x_global.end());
    thrust::universal_vector<double> d_yg(N_cx, 0.0);
    S_global.action(d_xg.data().get(), d_yg.data().get());
    cudaDeviceSynchronize();

    std::vector<double> y_ref(2 * n_local);
    for (int k = 0; k < n_local; ++k)
    {
        const int g = gI(k, 0);
        y_ref[k] = d_yg[g];
        y_ref[n_local + k] = d_yg[ndof + g];
    }

    double err = 0.0, ref_norm = 0.0;
    for (int i = 0; i < 2 * n_local; ++i)
    {
        err = std::max(err, std::abs(y_block[i] - y_ref[i]));
        ref_norm = std::max(ref_norm, std::abs(y_ref[i]));
    }
    const double rel = (ref_norm > 0.0) ? err / ref_norm : err;

    if (rel < 1e-10)
        log.pass("BlockSparseMatrix single-subdomain assembly matches global Helmholtz (constant a=1.5)");
    else
        log.fail("BlockSparseMatrix single-subdomain assembly matches global Helmholtz (constant a=1.5)",
                 std::format("L∞ rel err = {:.2e}", rel));
}

int main()
{
    TestLogger log;
    test_single_subdomain_assembly(log);
    test_single_subdomain_assembly_variable_a(log);
    return log.finish();
}
