/**
 * @file HelmholtzDirect3D.cpp
 * @brief Example driver for solving the 3D Helmholtz equation with a sparse direct solver
 *
 * @details This file solves the same problem as Helmholtz3D.cpp:
 *
 *      -div(grad u) - omega^2 a^2(x) u == f    in  D := [-1, 1]^3
 *      -i omega a(x) u + du/dn == 0            on boundary of D
 *
 * but replaces the iterative MINRES solver with a sparse direct LU factorization
 * via SparseLU (backed by CuDSS).
 *
 * The FEM operator A = S - omega^2*M - i*omega*H is assembled once into a
 * SparseMatrix<double, true> and then factored. Subsequent solves are O(nnz).
 *
 * Helmholtz3D::action computes conj(A*z), so iterative solvers need rhs conj(b).
 * SparseLU operates on the assembled A directly: lu.solve(b, x) solves A*x = b.
 *
 * To compile & run:
 *  (1) cmake . && make cuddh -j
 *  (2) make HelmholtzDirect3D
 *  (3) ./examples/HelmholtzDirect3D
 *
 * Output files in solution/ are identical in format to Helmholtz3D.cpp and can
 * be visualized with visualize.py.
 */

#include <format>

#include "CLI11.hpp"
#include "cuddh.hpp"
#include "examples.hpp"

using namespace cuddh;

/// @brief forcing term, approximate point source
__device__ static double f(double3 x, double omega)
{
    double s = omega * omega;

    double r1 = (x.x - 0.5) * (x.x - 0.5) + x.y * x.y + x.z * x.z;
    double F = std::pow(s / M_PI, 1.5) * std::exp(-s * r1);

    double r2 = (x.x + 0.7) * (x.x + 0.7) + (x.y + 0.7) * (x.y + 0.7) + x.z * x.z;
    F += std::pow(s / M_PI, 1.5) * std::exp(-s * r2);

    return F;
}

/// @brief alpha(x) = 1/c(x) where c(x) is the wave-speed.
__device__ static double alpha(double3 x)
{
    const double r = max(abs(x.x), abs(x.y));
    return (r < 0.5) ? 0.5 : 1.0;
}

int main(int argc, char *argv[])
{
    int deg = 3;
    std::vector<int> grid = {16};
    double omega = -1.0;

    CLI::App app{"HelmholtzDirect3D: Sparse direct solver for the 3D Helmholtz equation"};
    app.add_option("-p,--deg", deg, "Polynomial degree of basis functions")->default_val(3);
    app.add_option("-n,--grid", grid, "Grid dimensions: nx [ny [nz]] (if omitted, ny=nz=nx)")
        ->expected(1, 3)
        ->default_val("16");
    app.add_option("-w,--omega", omega, "Helmholtz frequency (default: 0.1 * nx * deg)");
    CLI11_PARSE(app, argc, argv);

    const int nx = grid[0];
    const int ny = grid.size() > 1 ? grid[1] : grid[0];
    const int nz = grid.size() > 2 ? grid[2] : grid[0];
    if (omega < 0.0)
        omega = 0.1 * nx * deg;

    // Assemble the mesh
    Mesh3D mesh = Mesh3D::uniform_cube(nx, -1.0, 1.0, ny, -1.0, 1.0, nz, -1.0, 1.0);

    // Construct 1D basis functions
    Basis basis(deg + 1);

    // FEM space
    H1Space3D fem(mesh, basis);
    const int ndof = fem.size();
    const int N = 2 * ndof;

    std::cout << "Solving the 3D Helmholtz equation with a sparse direct solver...\n"
              << "\tomega = " << omega << "\n"
              << "\t#elements = " << mesh.n_elem() << "\n"
              << "\tpolynomial degree = " << deg << "\n"
              << "\t#dof = " << N << "\n";

    // Boundary faces and trace space for Robin BCs
    auto boundary_faces = mesh.get_boundary_faces();
    TraceSpace3D fs(fem, boundary_faces.size(), boundary_faces);

    auto a = gridfunc(fem, [=] __device__(double3 x) -> double { return alpha(x); });

    Helmholtz3D A(fem, fs, omega, a);

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

    l2_project(MassMatrix3D(fem), [=] __device__(double3 x) -> double { return f(x, omega); }, b);

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
    auto coo = fem.physical_coordinates(MemorySpace::HOST);

    const char coo_file[] = "solution/coo.0000";
    const char sol_file[] = "solution/uv.0000";

    if (to_file(coo_file, coo.size(), coo.data()))
        std::cout << "Saved collocation points to " << coo_file << std::endl;
    else
        std::cerr << "Failed to save collocation points to " << coo_file << std::endl;

    if (to_file(sol_file, N, u))
        std::cout << "Saved solution to " << sol_file << std::endl;
    else
        std::cerr << "Failed to save solution to " << sol_file << std::endl;

    return 0;
}
