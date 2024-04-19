/**
 * @file Helmholtz.cpp
 * @brief Example driver for solving the Helmholtz equation
 *
 * @details This file is a driver for solving the Helmholtz equation with
 * approximate absorbing boundary conditions:
 *
 *      -div(grad U) - omega^2 U == f    in  D := [-1, 1]x[-1, 1]
 *      i omega U + dU/dn == 0           on boundary of D
 *
 * Here omega is the frequency. We assume f is real valued, and U is complex
 * valued.
 *
 * Write U = u + i v, the weak formulation is
 *
 *      a([u, v], phi) == b(phi)        for all phi in H1(D)
 *
 * The bilinear form a is defined as
 *
 *      a([u, v], phi) = [ (grad u, grad phi) - omega^2 (u, phi) - omega <v, phi>;
 *                         (grad v, grad phi) - omega^2 (v, phi) + omega <u, phi> ]
 *
 * And the linear operator b is defined b(phi) = [ (f, phi); 0 ]
 * 
 * In cuddh, a is assembled from the following operators:
 *      
 *      StiffnessMatrix S;
 *      S.action(c, x, y); // y[i] <- y[i] + c * (grad x, grad phi[i]) where phi[i] is the i-th basis function
 *      
 *      MassMatrix M;
 *      M.action(c, x, y); // y[i] <- y[i] + c * (x, phi[i])
 *      
 *      FaceMassMatrix H;
 *      H.action(c, x, y); // y[i] <- y[i] + c * <x, phi[i]>
 * 
 * The bilinear form combines these operations in the Helmholtz class.
 * 
 * To compile & run this program:
 *  (1) From the CuDDHelmholtz directory, compile the library:
 *      cmake .
 *      make cuddh -j
 *  (2) compile the program:
 *      make Helmholtz
 *  (3) run:
 *      ./examples/Helmholtz
 *
 * The program will write the collocation points to `solution/xy.0000` in binary
 * format. The solution is written to `solution/helmholtz.0000` in binary
 * format.
 *
 * This format can be read and visualized, for example, in Python using numpy and matplotlib via:
 *      
 *      xy = numpy.fromfile("solution/xy.0000", order='F')
 *      xy = xy.reshape(2, -1)
 *      x, y = xy[0], xy[1]
 *
 *      uv = numpy.fromfile("solution/helmholtz.0000", order='F')
 *      uv = uv.reshape(-1, 2)
 *      U  = uv[:, 0] + 1j * uv[:, 1]
 * 
 *      # visualize the modulus of U
 *      matplotlib.pyplot.tricontourf(x, y, np.abs(U))
 */

#include "cuddh.hpp"
#include "examples.hpp"

using namespace cuddh;

/// @brief FEM discretization of the Helmholtz equation -div(grad u) - omega^2 u == f
/// with boundary conditions: du/dn + i omega u == 0.
class Helmholtz : public Operator
{
public:
    Helmholtz(double omega, const H1Space& fem, const FaceSpace& fs);

    /// @brief y[i] = a(x, phi[i]) where a(u,v) = (grad u, grad v) - omega^2 (u, v) - i*omega <u, v> 
    /// @param x the real and imaginary part of the solution
    /// @param y on exit y[i] <- a(x, v[i]) for each v
    void action(const double * x, double * y) const;

    /// @brief y[i] <- y[i] + c * a(x, phi[i]) where a(u,v) = (grad u, grad v) - omega^2 (u, v) - i*omega <u, v> 
    /// @param c
    /// @param x the real and imaginary part of the solution
    /// @param y on exit y[i] <- a(x, v[i]) for each v
    void action(double c, const double * x, double * y) const;

private:
    const double omega;
    const int ndof;
    
    const H1Space& fem;
    const FaceSpace& fs;
    
    StiffnessMatrix S;
    MassMatrix M;
    FaceMassMatrix H;
    
    mutable dvec xf;
    mutable dvec yf;
};

/// @brief forcing term
static double f(const double X[2])
{
    const double x = X[0], y = X[1];
    double r = x * x + y * y;
    return 20 * std::exp(-400 * r);
}

int main()
{
    const double omega = 10.0; // Helmholtz frequency
    const int deg = 3; // polynomial degree of basis functions
    const int nx = 15; // number of elements along each direction. Mesh will have nx^2 elements
    
    const int m = 200; // number of vectors in the Krylov space used in each iteration of GMRES
    const int maxit = 20; // maximum number of iterations of GMRES
    const double tol = 1e-6; // relative tolerance. GMRES stops when ||b-A*x|| < tol*||b||
    
    // Assemble the mesh
    Mesh2D mesh = Mesh2D::uniform_rect(nx, -1.0, 1.0, nx, -1.0, 1.0);

    // Construct 1D basis functions. On each element, the 2D basis functions are
    // tensor products of these 1D basis functions.
    Basis basis(deg+1);

    // The mesh and 1D basis functions are combined in H1Space to define the
    // total global degrees of freedom of the problem.
    H1Space fem(mesh, basis);
    const int ndof = fem.size(); // # of degrees of freedom

    std::cout << "Solving the Helmholtz equation...\n"
              << "\tomega = " << omega << "\n"
              << "\t#elements = " << mesh.n_elem() << "\n"
              << "\tpolynomial degree = " << deg << "\n"
              << "\t#dof = " << ndof << "\n"
              << "\tgmres m = " << m << "\n";

    // identify the boundary faces in the mesh in order to define the FaceSpace
    // and FaceMassMatrix
    ivec boundary_faces = mesh.boundary_edges();

    // The FaceSpace is a subspace of the H1Space used to identify the degrees
    // of freedom needed in the computation of trace terms: <u, phi>
    FaceSpace fs(fem, boundary_faces.size(), boundary_faces);

    const int N = 2 * ndof; // total degrees of freedom in [u, v] (U := u + i v)

    dvec U(N); // the solution vector [u; v]
    dvec b(N); // the right hand side b(phi)

    LinearFunctional l(fem); // computes projections like (f, phi)
    dvec_wrapper bu(b.data(), ndof); // bu <- (f, phi)
    l.action(1.0, f, bu); // bu[i] <- (f, phi[i])

    // The operator representing the bilinear form: a([u, v], phi)
    Helmholtz A(omega, fem, fs);

    // dummy preconditioner
    Identity Id(N);

    // solve a([u, v], phi) = b(phi)
    auto out = gmres(N, U, &A, b, &Id, m, maxit, tol, 1);

    // save solution and collocation nodes to file
    auto xy = fem.physical_coordinates();
    to_file("solution/xy.0000", N, xy);
    to_file("solution/helmholtz.0000", N, U);

    return 0;
}

Helmholtz::Helmholtz(double w, const H1Space& fem_, const FaceSpace& fs_)
    : omega{w},
      ndof{fem_.size()},
      fem{fem_},
      fs{fs_},
      S(fem),
      M(fem),
      H(fs),
      xf(fs.size()),
      yf(fs.size()) {}

void Helmholtz::action(double c, const double * x, double * y) const
{
    const double * u = x;
    const double * v = x + ndof;
    double * Au = y;
    double * Av = y + ndof;
    
    S.action(1.0, u, Au);
    S.action(1.0, v, Av);

    M.action(-omega*omega, u, Au);
    M.action(-omega*omega, v, Av);

    yf.zeros();
    fs.restrict(v, xf);
    H.action(-omega, xf, yf);
    fs.prolong(yf, Au);

    yf.zeros();
    fs.restrict(u, xf);
    H.action(omega, xf, yf);
    fs.prolong(yf, Av);
}

void Helmholtz::action(const double * x, double * y) const
{
    const int N = 2 * ndof;
    for (int i = 0; i < N; ++i)
        y[i] = 0.0;
    action(1.0, x, y);
}
