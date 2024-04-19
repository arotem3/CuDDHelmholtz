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
class Poisson : public Operator
{
public:
    Poisson(const H1Space& fem, const FaceSpace& fs);

    void action(const double * x, double * y) const;

    void action(double c, const double * x, double * y) const;

private:
    const int ndof;
    StiffnessMatrix a;
    H1_0 pr;
};

static double f(const double X[2])
{
    return 1.0;
}

static double g(const double X[2])
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
    
    const int gmres_m = 20; // number of vectors in the Krylov space used in each iteration of GMRES
    const int gmres_maxit = 20; // maximum number of iterations of GMRES
    const double gmres_tol = 1e-6; // relative tolerance. GMRES stops when ||b-A*x|| < tol*||b||
    
    // Assemble the mesh
    Mesh2D mesh = Mesh2D::uniform_rect(nx, -1.0, 1.0, nx, -1.0, 1.0);

    // Construct 1D basis functions. On each element, the 2D basis functions are
    // tensor products of these 1D basis functions.
    Basis basis(deg+1);

    // The mesh and 1D basis functions are combined in H1Space to define the
    // total global degrees of freedom of the problem.
    H1Space fem(mesh, basis);
    const int ndof = fem.size(); // # of degrees of freedom

    std::cout << "Solving the Poisson equation...\n"
              << "\t#elements = " << mesh.n_elem() << "\n"
              << "\tpolynomial degree = " << deg << "\n"
              << "\t#dof = " << ndof << "\n"
              << "\tgmres m = " << gmres_m << "\n";

    // identify the boundary faces in the mesh in order to define the FaceSpace
    // and FaceMassMatrix
    ivec boundary_faces = mesh.boundary_edges();

    // The FaceSpace is a subspace of the H1Space used to identify the degrees
    // of freedom needed in the computation of trace terms: <u, phi>
    FaceSpace fs(fem, boundary_faces.size(), boundary_faces);
    const int fdof = fs.size();

    dvec u(ndof); // the solution vector
    dvec b(ndof); // the right hand side: (f, phi) - (grad g, grad phi)

    // linear system
    Poisson A(fem, fs);

    // set up right hand side
    LinearFunctional l(fem);
    l.action(1.0, f, b); // (f, phi)

    FaceLinearFunctional fl(fs);
    dvec q(fdof);
    dvec y(fdof); // (g, phi)
    fl.action(1.0, g, y);

    FaceMassMatrix m(fs);
    DiagInvFaceMassMatrix p(fs);
    auto out = gmres(fdof, q, &m, y, &p, 10, 10, 1e-12);

    dvec G(ndof);
    fs.prolong(q, G); // G

    A.action(-1.0, G, b); // b <- b + (grad G, grad phi)

    // dummy preconditioner
    Identity Id(ndof);

    // solve for u
    out = gmres(ndof, u, &A, b, &Id, gmres_m, gmres_maxit, gmres_tol, 1);

    // add G to u
    for (int i = 0; i < ndof; ++i)
        u[i] += G[i];

    // save solution and collocation nodes to file
    auto xy = fem.physical_coordinates();
    to_file("solution/xy.0000", 2*ndof, xy);
    to_file("solution/poisson.0000", ndof, u);

    return 0;
}

Poisson::Poisson(const H1Space& fem, const FaceSpace& fs)
    : ndof{fem.size()},
      a(fem),
      pr(fs) {}

void Poisson::action(double c, const double * x, double * y) const
{
    a.action(c, x, y);
    pr.action(y);
}

void Poisson::action(const double * x, double * y) const
{
    a.action(x, y);
    pr.action(y);
}