/**
 * @file HelmholtzDirect.cpp
 * @brief Example driver for solving the Helmholtz equation with a sparse direct solver
 *
 * @details This file solves the same problem as Helmholtz.cpp:
 *
 *      -div(grad u) - omega^2 a^2(x) u == f    in  D := [-1, 1]x[-1, 1]
 *      i a(x) omega u + du/dn == 0             on boundary of D
 *
 * but replaces the iterative MINRES solver with a sparse direct LU factorization
 * via SparseLU (backed by CuDSS).
 *
 * The FEM operator A = S - omega^2*M - i*omega*H is assembled once into a
 * SparseMatrix<double, true> and then factored. Subsequent solves are O(nnz).
 *
 * Helmholtz::action computes conj(A*z), so iterative solvers need rhs conj(b).
 * SparseLU operates on the assembled A directly: lu.solve(b, x) solves A*x = b.
 *
 * To compile & run:
 *  (1) cmake . && make cuddh -j
 *  (2) make HelmholtzDirect
 *  (3) ./examples/HelmholtzDirect
 *
 * Output files in solution/ are identical in format to Helmholtz.cpp and can
 * be visualized with visualize.py.
 */

#include "CLI11.hpp"
#include "cuddh.hpp"
#include "examples.hpp"

using namespace cuddh;

/// @brief forcing term, approximate point source
__device__ static double f(const double2 X, double omega)
{
    const auto [x, y] = X;
    double s = omega * omega;

    double r = (x + 0.5) * (x + 0.5) + y * y;
    double F = s / M_PI * std::exp(-s * r);

    r = (x - 0.5) * (x - 0.5) + (y + 0.5) * (y + 0.5);
    F += s / M_PI * std::exp(-s * r);
    return F;
}

/// @brief a(x) = 1/c(x) where c(x) is the wave-speed.
__device__ static double a(const double2 X)
{
    const auto [x, y] = X;
    const double r = max(abs(x), abs(y));
    return (r < 0.5) ? 0.5 : 1.0;
}

int main(int argc, char *argv[])
{
    int deg = 3;
    std::vector<int> grid = {32};
    double omega = -1.0;

    CLI::App app{"HelmholtzDirect: Sparse direct solver for the 2D Helmholtz equation"};
    app.add_option("-p,--deg", deg, "Polynomial degree of basis functions")->default_val(3);
    app.add_option("-n,--grid", grid, "Grid dimensions: nx [ny] (if ny omitted, ny=nx)")
        ->expected(1, 2)
        ->default_val("32");
    app.add_option("-w,--omega", omega, "Helmholtz frequency (default: 0.1 * nx * deg)");
    CLI11_PARSE(app, argc, argv);

    const int nx = grid[0];
    const int ny = grid.size() > 1 ? grid[1] : grid[0];
    if (omega < 0.0)
        omega = 0.1 * nx * deg;

    // Assemble the mesh
    Mesh2D mesh = Mesh2D::uniform_rect(nx, -1.0, 1.0, ny, -1.0, 1.0);

    // Construct 1D basis functions
    Basis basis(deg + 1);

    // FEM space
    H1Space2D fem(mesh, basis);
    const int ndof = fem.size();
    const int N = 2 * ndof;

    // Boundary trace space for Robin BCs
    ivec boundary_faces = mesh.boundary_edges();
    TraceSpace2D fs(fem, boundary_faces.size(), boundary_faces);

    auto coef = gridfunc(fem, [] __device__(double2 x) -> double { return a(x); });

    std::cout << "Solving the Helmholtz equation with a sparse direct solver...\n"
              << "\tomega = " << omega << "\n"
              << "\t#elements = " << mesh.n_elem() << "\n"
              << "\tpolynomial degree = " << deg << "\n"
              << "\t#dof = " << N << "\n";

    Helmholtz A(fem, fs, omega, coef);

    // Build sparsity pattern from FEM connectivity
    SparseMatrix<double, true> S(ndof, ndof, 0, SparseMatrixType::Symmetric);
    fem.set_pattern(S);
    S.finalize_pattern();
    std::cout << "\n\t#nnz = " << S.nnz() << "\n";

    // Assemble A = S - omega^2*M - i*omega*H into the sparse matrix
    A.assemble(std::complex<double>(1.0, 0.0), S);
    S.finalize_values();

    // Factorize (analysis + LU in constructor)
    std::cout << "\nfactorizing... \n";
    SparseLU<double, true> lu(S);

    // Right-hand side b = M*f, projected onto the FEM space
    thrust::universal_vector<double> B(N, 0.0), U(N, 0.0);
    double *b = thrust::raw_pointer_cast(B.data());
    double *u = thrust::raw_pointer_cast(U.data());

    l2_project(b, MassMatrix(fem), [=] __device__(const double2 X) -> double { return f(X, omega); });

    // SparseLU solves A*x = b directly; no conjugation of b or x is needed.
    lu.solve(b, u);
    CUDDH_CUDA_CHECK(cudaDeviceSynchronize());

    // Relative residual: ||action(x) - b|| / ||b||
    thrust::universal_vector<double> R(N, 0.0);
    double *r = thrust::raw_pointer_cast(R.data());
    A.action(u, r);
    std::cout << std::format("relative residual: {:.3e}\n", dla::dist(N, r, b) / dla::norm(N, b));

    lu.print(std::cout);

    // Save solution and collocation nodes to file
    auto xy = fem.physical_coordinates(MemorySpace::HOST);

    const char xy_file[] = "solution/xy.0000";
    const char sol_file[] = "solution/uv.0000";

    if (to_file(xy_file, xy.size(), xy.data()))
        std::cout << "Saved collocation points to " << xy_file << std::endl;
    else
        std::cerr << "Failed to save collocation points to " << xy_file << std::endl;

    if (to_file(sol_file, N, u))
        std::cout << "Saved solution to " << sol_file << std::endl;
    else
        std::cerr << "Failed to save solution to " << sol_file << std::endl;

    return 0;
}
