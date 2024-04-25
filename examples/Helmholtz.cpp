/**
 * @file Helmholtz.cpp
 * @brief Example driver for solving the Helmholtz equation
 *
 * @details This file is a driver for solving the Helmholtz equation with
 * approximate absorbing boundary conditions:
 *
 *      -div(grad U) - omega^2 a^2(x) U == f    in  D := [-1, 1]x[-1, 1]
 *      i a(x) omega U + dU/dn == 0             on boundary of D
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
 *      a([u, v], phi) = [ (grad u, grad phi) - omega^2 (a^2(x) u, phi) - omega <a(x) v, phi>;
 *                         (grad v, grad phi) - omega^2 (a^2(x) v, phi) + omega <a(x) u, phi> ]
 *
 * And the linear operator b is defined b(phi) = [ (f, phi); 0 ]
 * 
 * In cuddh, a is assembled from the following operators:
 *      
 *      StiffnessMatrix S;
 *      S.action(c, x, y); // y[i] <- y[i] + c * (grad x, grad phi[i]) where phi[i] is the i-th basis function
 *      
 *      MassMatrix M;
 *      M.action(c, x, y); // y[i] <- y[i] + c * (a(x)^2 x, phi[i])
 *      
 *      FaceMassMatrix H;
 *      H.action(c, x, y); // y[i] <- y[i] + c * <a(x)  x, phi[i]>
 * 
 * The Helmholtz class combines these operations to define the bilinear form a(*,*).
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
    Helmholtz(double omega, const double * a2x, const double * ax, const H1Space& fem, const FaceSpace& fs);

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
    const int fdof;
    
    const H1Space& fem;
    const FaceSpace& fs;
    
    StiffnessMatrix S;
    MassMatrix M;
    FaceMassMatrix H;
    
    mutable host_device_dvec xf;
    mutable host_device_dvec yf;
};

/// @brief forcing term, approximate point source
__device__ static double f(const double X[2])
{
    constexpr double s = 20; // bigger s --> more localized
    constexpr double sqrt_pi = 1.772453850905516;

    const double x = X[0]-0.25, y = X[1];
    double r = x * x + y * y;
    return (s/sqrt_pi) * std::exp(-(s * s) * r);
}

/// @brief a(x) = 1/c(x) where c(x) is the wave-speed.
__device__ static double a(const double X[2])
{
    const double x = X[0] + 0.25, y = X[1] + 0.25;
    double r = x*x + y*y;
    r = std::pow(10 * r, 3);
    return 1.0 - 0.8 * std::exp(-r);
}

/// @brief projects the coefficient a^2(x) onto the H1Space and a(x) onto the FaceSpace
static void project_coefficients(const H1Space& fem, const FaceSpace& fs,  double * a2x, double * ax);

int main()
{
    const double omega = 10.0; // Helmholtz frequency
    const int deg = 3; // polynomial degree of basis functions
    const int nx = 16; // number of elements along each direction. Mesh will have nx^2 elements
    
    const int m = 200; // number of vectors in the Krylov space used in each iteration of GMRES
    const int maxit = 100; // maximum number of iterations of GMRES
    const double tol = 1e-6; // relative tolerance. GMRES stops when ||b-A*x|| < tol*||b||
    const int verbose = 1; // 0: silent, 1: progress bar, 2: one line per iteration
    
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
              << "\t#dof = " << 2 * ndof << "\n";

    // identify the boundary faces in the mesh in order to define the FaceSpace
    // and FaceMassMatrix
    ivec boundary_faces = mesh.boundary_edges();

    // The FaceSpace is a subspace of the H1Space used to identify the degrees
    // of freedom needed in the computation of trace terms: <u, phi>
    FaceSpace fs(fem, boundary_faces.size(), boundary_faces);
    const int fdof = fs.size();

    const int N = 2 * ndof; // total degrees of freedom in [u, v] (U := u + i v)

    // To manage memory between host and device, we use the HostDeviceArray class.
    host_device_dvec U(N);
    host_device_dvec b(N);
    host_device_dvec a2x(ndof);

    host_device_dvec ax(fdof);
    
    double * d_U = U.device_write(); // the solution vector [u; v]
    double * d_b = b.device_write(); // the right hand side b(phi)
    double * d_a2 = a2x.device_write(); // a^2(x) projected onto H1Space
    
    double * d_a = ax.device_write(); // a(x) projected onto FaceSpace
    
    project_coefficients(fem, fs, d_a2, d_a);

    LinearFunctional l(fem); // computes integrals: (f, phi)
    l.action([] __device__ (const double X[2]) -> double {return f(X);}, d_b); // b[i] <- (f, phi[i])

    // The operator representing the bilinear form: a([u, v], phi)
    Helmholtz A(omega, d_a2, d_a, fem, fs);

    // solve a([u, v], phi) = b(phi)
    std::cout << "\nsolving with GMRES(" << m << ") ... \n";
    auto out = gmres(N, d_U, &A, d_b, m, maxit, tol, verbose);

    // copy solution to host
    const double * h_U = U.host_read();

    // save solution and collocation nodes to file
    auto xy = fem.physical_coordinates(MemorySpace::HOST);

    const char xy_file[] = "solution/xy.0000";
    const char sol_file[] = "solution/helmholtz.0000";
    to_file(xy_file, N, xy);
    to_file(sol_file, N, h_U);

    std::cout << "\nSolution written to: " << sol_file
              << "\nCoordinates written to: " << xy_file << "\n";

    return 0;
}

Helmholtz::Helmholtz(double w, const double * a2x, const double * ax, const H1Space& fem_, const FaceSpace& fs_)
    : omega{w},
      ndof{fem_.size()},
      fdof{fs_.size()},
      fem{fem_},
      fs{fs_},
      S(fem_),
      M(a2x, fem_),
      H(ax, fs_),
      xf(fdof),
      yf(fdof) {}

void Helmholtz::action(double c, const double * x, double * y) const
{
    cuddh_error("Helmholtz::action(c, x, y) not implemented");
}

void Helmholtz::action(const double * x, double * y) const
{
    const double * u = x;
    const double * v = x + ndof;

    double * Au = y;
    double * Av = y + ndof;

    double * d_yf = yf.device_write();
    double * d_xf = xf.device_write();

    S.action(u, Au);
    S.action(v, Av);

    M.action(-omega * omega, u, Au);
    M.action(-omega * omega, v, Av);

    zeros(fdof, d_yf);
    fs.restrict(v, d_xf);
    H.action(-omega, d_xf, d_yf);
    fs.prolong(d_yf, Au);

    zeros(fdof, d_yf);
    fs.restrict(u, d_xf);
    H.action(omega, d_xf, d_yf);
    fs.prolong(d_yf, Av);
}

void project_coefficients(const H1Space& fem, const FaceSpace& fs,  double * a2x, double * ax)
{
    int n_basis = fem.basis().size();
    QuadratureRule q(2 * n_basis, QuadratureRule::GaussLegendre);

    int ndof = fem.size();
    host_device_dvec pa2(ndof);

    double * d_pa2 = pa2.device_write();
    
    MassMatrix m(fem);
    DiagInvMassMatrix mi(fem);
    LinearFunctional l(fem, q);

    l.action([] __device__ (const double X[2]) -> double {double aX = a(X); return aX*aX;}, d_pa2);
    gmres(ndof, a2x, &m, d_pa2, &mi, 5, 10, 1e-12);

    int fdof = fs.size();
    host_device_dvec pa(fdof);

    double * d_pa = pa.device_write();

    FaceMassMatrix fm(fs);
    DiagInvFaceMassMatrix fmi(fs);
    FaceLinearFunctional fl(fs, q);

    fl.action([] __device__ (const double X[2]) -> double {return a(X);}, d_pa);
    gmres(fdof, ax, &fm, d_pa, &fmi, 5, 10, 1e-12);
}