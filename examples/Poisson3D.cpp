/**
 * @file Poisson3D.cpp
 * @brief Example driver for solving the Poisson equation in 3D
 *
 * @details This file is a driver for solving the Poisson equation with
 * Dirichlet boundary conditions:
 *
 *      -div(grad u) == f       in D := [-1, 1]^3
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
 *      x = numpy.fromfile("solution/x.0000", order='F')
 *      x = x.reshape(3, -1)
 *      x, y, z = x[0], x[1], x[2]
 *
 *      u = numpy.fromfile("solution/poisson.0000")
 * 
 *      
 */

#include "cuddh.hpp"
#include "examples.hpp"

using namespace cuddh;

/// @brief Bilinear form (grad u, grad phi) where phi are in H1_0.
class Poisson3D : public Operator
{
public:
    Poisson3D(const H1Space3D& fem, const TraceSpace3D& fs);

    void action(const double * x, double * y) const;
    void action(double c, const double * x, double * y) const;

private:
    const TraceSpace3D& tr;
    StiffnessMatrix3D a;
};

__device__ static double f(double3 r)
{
    return -4.0;
}

__device__ static double g(double3 r)
{
    return 3.0 * r.x * r.x - r.y * r.y + r.z;
}

int main()
{
    const int deg = 3; // polynomial degree of basis functions
    const int nx = 10; // number of elements in each direction. Mesh will have nx^3 elements

    const int gmres_m = 20; // number of vectors in the Krylov space used in each iteration of GMRES
    const int gmres_maxit = 100; // maximum number of iterations of GMRES
    const double gmres_tol = 1e-10; // relative tolerance. GMRES stops when ||b-A*x|| < tol*||b||
    const int gmres_verbose = 1; // 0: silent, 1: progress bar, 2: one line per iteration

    // create the mesh
    Mesh3D mesh = Mesh3D::uniform_cube(nx, -1.0, 1.0, nx, -1.0, 1.0, nx, -1.0, 1.0);

    // Construct 1D basis functions. On each element, the 3D basis functions are
    // tensor products of these 1D basis functions.
    Basis basis(deg+1);

    // The mesh and 1D basis functions are combined in H1Space3D to define the
    // total global degrees of freedom of the problem.
    H1Space3D fem(mesh, basis);
    const int ndof = fem.size(); // # of degrees of freedom

    std::cout << "Solving the Poisson equation...\n"
              << "\t#elements = " << mesh.n_elem() << "\n"
              << "\tpolynomial degree = " << deg << "\n"
              << "\t#dof = " << ndof << "\n";

    // identify the boundary faces in the mesh in order to define the TraceSpace3D
    // and FaceMassMatrix
    auto boundary_faces = mesh.get_boundary_faces();
    
    // The TraceSpace3D is a subspace of the H1Space3D used to identify the degrees
    // of freedom needed on the boundary of the domain. In particular, we use
    // this object to restrict the solution to H1_0.
    TraceSpace3D tr(fem, boundary_faces.size(), boundary_faces);
    const int fdof = tr.size();

    // To manage memory between host and device, we use the HostDeviceArray class.
    host_device_dvec _u(ndof); // the solution vector
    host_device_dvec _b(ndof); // the right hand side: (f, phi) - (grad g, grad phi)
    host_device_dvec _G(ndof); // the extension of g to H1

    host_device_dvec _q(fdof); // projection of g onto face space

    double * u = _u.device_write();
    double * b = _b.device_write();
    double * q = _q.device_write();
    double * G = _G.device_write();

    // linear system
    Poisson3D A(fem, tr);

    // set the right hand side
    MassMatrix3D M(fem);
    l2_project(M, [] __device__ (double3 r) { return f(r); }, b); // (f, phi)
    tr.orth(b); // zero out the boundary terms

    trace(tr, [] __device__ (double3 r) { return g(r); }, q); // evaluate on the boundary

    tr.prolong(q, G); // extend q to H1
    A.action(-1.0, G, b); // b = b - (grad G, grad phi)

    // solve the linear system
    std::cout << "\nsolving with gmres(" << gmres_m << ")...\n";
    auto out = gmres(ndof, u, &A, b, gmres_m, gmres_maxit, gmres_tol, gmres_verbose);

    // add G to u
    axpby(ndof, 1.0, G, 1.0, u);

    // compare against exact
    host_device_dvec _ue(ndof);
    double * ue = _ue.device_write();
    auto x = fem.physical_coordinates(MemorySpace::DEVICE);
    forall(ndof, [=] __device__ (int i) { ue[i] = g(x[i]); });
    std::cout << "\nL2 error ~ " << l2_dist(M, u, ue) << "\n\n";

    // copy to host
    const double *h_u = _u.host_read();
    x = fem.physical_coordinates(MemorySpace::HOST);

    // save solution
    const char x_file[] = "solution/x.0000";
    const char u_file[] = "solution/poisson.0000";

    if (to_file(x_file, ndof, x.data()))
        std::cout << "Coordinates written to " << x_file << "\n";
    if (to_file(u_file, ndof, h_u))
        std::cout << "Solution written to " << u_file << "\n";
    
    return 0;
}

Poisson3D::Poisson3D(const H1Space3D& fem, const TraceSpace3D& tr)
    : tr(tr), a(fem)
{
}

void Poisson3D::action(const double * x, double * y) const
{
    a.action(x, y);
    tr.orth(y);
}

void Poisson3D::action(double c, const double * x, double * y) const
{
    a.action(c, x, y);
    tr.orth(y);
}