/**
 * @file Poisson.cpp
 * @brief Example driver for solving the Poisson equation
 *
 * @details This file is a driver for solving the Poisson equation with
 * Dirichlet boundary conditions:
 *
 *      -div(grad u) == f       in D := [-1, 1]x[-1, 1]
 *          u == g              on boundary of D
 *
 * Define u = v + g where g is an extension of the Dirichlet data to all of D.
 * The weak formulation is
 *
 *      a(v, phi) == b(phi)     for all phi in H1_0(D)
 *
 * The bilinear form a is defined as
 *
 *      a(v, phi) = (grad v, grad phi)
 *
 * And the linear operator b is defined as b(phi) = (f, phi) - (grad g, grad phi),
 *
 * In cuddh, a is computed via StiffnessMatrix::action.
 *
 * To compile & run this program:
 *  (1) From the CuDDHelmholtz directory, compile the library:
 *      cmake .
 *      make cuddh -j
 *  (2) compile the program:
 *      make Poisson
 *  (3) run:
 *      ./examples/Poisson
 *
 * The program will write the collocation points to `solution/xy.0000` in binary
 * format. The solution is written to `solution/poisson.0000` in binary
 * format.
 *
 * This format can be read and visualized, for example, in Python using numpy and matplotlib via:
 *
 *      xy = numpy.fromfile("solution/xy.0000", order='F')
 *      xy = xy.reshape(2, -1)
 *      x, y = xy[0], xy[1]
 *
 *      u = numpy.fromfile("solution/poisson.0000")
 *
 *      matplotlib.pyplot.tricontourf(x, y, u)
 */

#include "cuddh.hpp"
#include "examples.hpp"

using namespace cuddh;

/// @brief Bilinear form (grad u, grad phi) where phi are in H1_0.
class Poisson : public Operator<double>
{
public:
    Poisson(const H1Space2D &fem, const TraceSpace2D &fs);

    void action(const double *x, double *y) const;

    void action(double c, const double *x, double *y) const;

private:
    StiffnessMatrix a;
    const TraceSpace2D &fs;
};

__device__ static double f(const double X[2])
{
    return 1.0;
}

__device__ static double g(const double X[2])
{
    const double x = X[0], y = X[1];
    if (std::abs(x - 1.0) < 1e-12)
        return 1.0 - y * y;
    else if (std::abs(x + 1.0) < 1e-12)
        return y * (1.0 - y * y);
    return 0.0;
}

int main()
{
    const int deg = 3; // polynomial degree of basis functions
    const int nx = 15; // number of elements along each direction. Mesh will have nx^2 elements

    SolverParams opts = {
        .maxit = 1000,                       // maximum number of minres iterations
        .rtol = 1e-6,                        // relative tolerance. minres stops when ||b-A*x|| < tol*||b||
        .verbose = SolverParams::ProgressBar // Silent, ProgressBar, Iterations
    };

    // Assemble the mesh
    Mesh2D mesh = Mesh2D::uniform_rect(nx, -1.0, 1.0, nx, -1.0, 1.0);

    // Construct 1D basis functions. On each element, the 2D basis functions are
    // tensor products of these 1D basis functions.
    Basis basis(deg + 1);

    // The mesh and 1D basis functions are combined in H1Space2D to define the
    // total global degrees of freedom of the problem.
    H1Space2D fem(mesh, basis);
    const int ndof = fem.size(); // # of degrees of freedom

    std::cout << "Solving the Poisson equation...\n"
              << "\t#elements = " << mesh.n_elem() << "\n"
              << "\tpolynomial degree = " << deg << "\n"
              << "\t#dof = " << ndof << "\n";

    // identify the boundary faces in the mesh in order to define the TraceSpace2D
    // and FaceMassMatrix
    ivec boundary_faces = mesh.boundary_edges();

    // The TraceSpace2D is a subspace of the H1Space2D used to identify the degrees
    // of freedom needed on the boundary of the domain. In particular, we use
    // this object to restrict the solution to H1_0.
    TraceSpace2D fs(fem, boundary_faces.size(), boundary_faces);
    const int fdof = fs.size();

    // To manage memory between host and device, we use the HostDeviceArray class.
    host_device_dvec _u(ndof);
    host_device_dvec _b(ndof);
    host_device_dvec _G(ndof);

    double *u = _u.device_write(); // the solution vector
    double *b = _b.device_write(); // the right hand side: (f, phi) - (grad g, grad phi)
    double *G = _G.device_write(); // the extension of q to H1

    // linear system
    Poisson A(fem, fs);

    // set up right hand side
    l2_project(b, MassMatrix(fem), [] __device__(const double X[2]) -> double { return f(X); }); // (f, phi)
    fs.orth(b); // zero out boundary terms

    // evaluate g on the boundary faces
    auto _q = trace(fs, [] __device__(const double X[2]) -> double { return g(X); });
    double *q = thrust::raw_pointer_cast(_q.data());

    fs.prolong(q, G);     // extend q to H1
    A.action(-1.0, G, b); // b <- b - (grad G, grad phi)

    // solve for u (without boundary conditions)
    std::cout << "\nsolving with minres... \n";
    auto out = minres(u, A, b, opts);

    // add G to u (now u satisfies Dirichlet BCs)
    dla::axpby(ndof, 1.0, G, 1.0, u); // u <- u + G

    // copy to host
    const double *h_u = _u.host_read();

    // save solution and collocation nodes to file
    auto xy = fem.physical_coordinates(MemorySpace::HOST);

    const char xy_file[] = "solution/xy.0000";
    const char sol_file[] = "solution/poisson.0000";
    const char res_file[] = "solution/residuals.0000";

    if (to_file(xy_file, 2 * ndof, xy.data()))
        std::cout << "Coordinates written to: " << xy_file << "\n";
    else
        std::cerr << "Failed to write coordinates to: " << xy_file << "\n";

    if (to_file(sol_file, ndof, h_u))
        std::cout << "Solution written to: " << sol_file << "\n";
    else
        std::cerr << "Failed to write solution to: " << sol_file << "\n";

    if (to_file(res_file, out.res_norm.size(), out.res_norm.data()))
        std::cout << "Residuals written to: " << res_file << "\n";
    else
        std::cerr << "Failed to write residuals to: " << res_file << "\n";

    return 0;
}

Poisson::Poisson(const H1Space2D &fem, const TraceSpace2D &fs_) : Operator<double>(fem.size()), a(fem), fs{fs_} {}

void Poisson::action(double c, const double *x, double *y) const
{
    a.action(c, x, y);
    fs.orth(y);
}

void Poisson::action(const double *x, double *y) const
{
    a.action(x, y);
    fs.orth(y);
}