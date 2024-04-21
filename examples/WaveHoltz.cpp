/**
 * @file WaveHoltz.cpp
 * @brief Example driver for solving the Helmholtz equation with WaveHoltz
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
 * To compile & run this program:
 *  (1) From the CuDDHelmholtz directory, compile the library:
 *      cmake .
 *      make cuddh -j
 *  (2) compile the program:
 *      make WaveHoltz
 *  (3) run:
 *      ./examples/WaveHoltz
 *
 * The program will write the collocation points to `solution/xy.00000` in binary
 * format. The solution is written to `solution/waveholtz.00000` in binary
 * format.
 *
 * This format can be read and visualized, for example, in Python using numpy and matplotlib via:
 *      
 *      xy = numpy.fromfile("solution/xy.0000", order='F')
 *      xy = xy.reshape(2, -1)
 *      x, y = xy[0], xy[1]
 *
 *      uv = numpy.fromfile("solution/waveholtz.0000", order='F')
 *      uv = uv.reshape(-1, 2)
 *      U  = uv[:, 0] + 1j * uv[:, 1]
 * 
 *      # visualize the modulus of U
 *      matplotlib.pyplot.tricontourf(x, y, np.abs(U))
 */

#include "cuddh.hpp"
#include "examples.hpp"

using namespace cuddh;

static double f(const double X[2])
{
    const double x = X[0], y = X[1];
    const double r = x * x + y * y;
    return 20 * std::exp(-400 * r);
}

class WaveHoltz : public Operator
{
public:
    WaveHoltz(double omega, const H1Space& fem, const FaceSpace& fs);

    /// @brief y <- (I - S) * x, where S is the linear part of the WaveHoltz operator 
    void action(const double * x, double * y) const override;

    /// @brief y <- y + c * (I - S) * x
    void action(double c, const double * x, double * y) const override;

    // computes the right hand side of the WaveHoltz system
    void force(double * f) const;

    int n_time_steps() const
    {
        return nt;
    }

    double K(double t) const
    {
        return (omega/M_PI) * (std::cos(omega * t) - 0.25);
    }

private:
    const int ndof;
    const int fdof;
    int nt;

    const double omega;
    double dt;

    const H1Space& fem;
    const FaceSpace& fs;

    StiffnessMatrix S;
    DiagInvMassMatrix Mi;
    FaceMassMatrix H;

    mutable dvec p;
    mutable dvec q;
    mutable dvec z;
    mutable dvec p_half;
    mutable dvec q_half;
    mutable dvec y0;
    mutable dvec y1;
};

int main()
{
    const double omega = 10.0; // Helmholtz frequency
    const int deg = 3; // polynomial degree of basis functions
    const int nx = 20; // number of elements along each direction. Mesh will have nx^2 elements
    
    const int gmres_m = 5; // number of vectors in the Krylov space used in each iteration of GMRES
    const int gmres_maxit = 100; // maximum number of iterations of GMRES
    const double gmres_tol = 1e-12; // relative tolerance. GMRES stops when ||b-A*x|| < tol*||b||
    
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
              << "\tgmres m = " << gmres_m << "\n";

    // identify the boundary faces in the mesh in order to define the FaceSpace
    // and FaceMassMatrix
    ivec boundary_faces = mesh.boundary_edges();

    // The FaceSpace is a subspace of the H1Space used to identify the degrees
    // of freedom needed in the computation of trace terms: <u, phi>
    FaceSpace fs(fem, boundary_faces.size(), boundary_faces);
    const int fdof = fs.size();

    // Assemble WaveHoltz op
    WaveHoltz A(omega, fem, fs);
    std::cout << "\t#time-steps/iteration = " << A.n_time_steps() << "\n";

    // Forcing
    dvec F(2 * ndof);
    LinearFunctional l(fem);
    l.action(1.0, f, F);

    A.force(F); // compute the right hand side of the WaveHoltz system

    // Solution vector U = [u; v]
    dvec U(2 * ndof);

    // dummy preconditioner
    Identity Id(2 * ndof);

    auto out = gmres(2*ndof, U, &A, F, &Id, gmres_m, gmres_maxit, gmres_tol, 1);

    // scale v by 1/omega to get imaginary part of physical solution
    for (int i = ndof; i < 2*ndof; ++i)
        U(i) /= omega;

    auto xy = fem.physical_coordinates();
    to_file("solution/xy.00000", 2*ndof, xy);
    to_file("solution/waveholtz.00000", 2*ndof, U);

    return 0;
}

WaveHoltz::WaveHoltz(double omega_, const H1Space& fem_, const FaceSpace& fs_)
    : ndof{fem_.size()},
      fdof{fs_.size()},
      omega{omega_},
      fem{fem_},
      fs{fs_},
      S(fem),
      Mi(fem),
      H(fs),
      p(ndof), q(ndof), z(ndof), p_half(ndof), q_half(ndof),
      y0(fdof), y1(fdof)
{
    double h = fem.mesh().min_h();
    int nb = fem.basis().size();
    dt = h / std::pow(nb, 2);
    double T = 2 * M_PI / omega;
    nt = std::ceil(T / dt);
    dt = T / nt;
}

void WaveHoltz::action(const double * x, double * y) const
{
    for (int i = 0; i < 2*ndof; ++i)
        y[i] = 0.0;
    action(1.0, x, y);
}

void WaveHoltz::action(double c, const double * x, double * y) const
{
    dvec_wrapper u(y, ndof);
    dvec_wrapper v(y+ndof, ndof);

    double dK = c * 0.5 * K(0) * dt;
    for (int i = 0; i < ndof; ++i)
    {
        p(i) = x[i];
        q(i) = x[ndof+i];

        u(i) += (c - dK) * p(i);
        v(i) += (c - dK) * q(i);
    }

    double t = 0.0;
    for (int it = 1; it < nt; ++it)
    {
        S.action(p, z); // z <- S * p

        fs.restrict(q, y0);
        y1.zeros();
        H.action(-1.0, y0, y1);
        fs.prolong(y1, z); // z <- z - H * q

        Mi.action(z, q_half);

        for (int i = 0; i < ndof; ++i)
        {
            p_half(i) = p(i) - 0.5 * dt * q(i);
            q_half(i) = q(i) + 0.5 * dt * q_half(i);
        }

        t += 0.5 * dt;

        S.action(p_half, z);

        fs.restrict(q_half, y0);
        y1.zeros();
        H.action(-1.0, y0, y1);
        fs.prolong(y1, z);

        Mi.action(z, p_half);

        t += 0.5 * dt;
        const double dK = c * dt * K(t) * ((it == nt) ? 0.5 : 1.0);
        for (int i = 0; i < ndof; ++i)
        {
            p(i) -= dt * q_half(i);
            q(i) += dt * p_half(i);

            u(i) -= dK * p(i);
            v(i) -= dK * q(i);
        }
    }
}

void WaveHoltz::force(double * f) const
{
    dvec_wrapper u(f, ndof);
    dvec_wrapper v(f+ndof, ndof);

    dvec F(ndof);

    for (int i = 0; i < ndof; ++i)
    {
        F(i) = f[i];
        
        p(i) = 0.0;
        q(i) = 0.0;

        u(i) = 0.0;
        v(i) = 0.0;
    }

    double t = 0.0;
    for (int it = 1; it < nt; ++it)
    {
        double c = std::cos(omega * t);

        S.action(p, z); // z <- S * p

        fs.restrict(q, y0);
        y1.zeros();
        H.action(-1.0, y0, y1);
        fs.prolong(y1, z); // z <- z - H * q

        for (int i = 0; i < ndof; ++i)
            z(i) -= c * F(i);

        Mi.action(z, q_half);

        for (int i = 0; i < ndof; ++i)
        {
            p_half(i) = p(i) - 0.5 * dt * q(i);
            q_half(i) = q(i) + 0.5 * dt * q_half(i);
        }

        t += 0.5 * dt;
        c = std::cos(omega * t);

        S.action(p_half, z);

        fs.restrict(q_half, y0);
        y1.zeros();
        H.action(-1.0, y0, y1);
        fs.prolong(y1, z);

        for (int i = 0; i < ndof; ++i)
            z(i) -= c * F(i);

        Mi.action(z, p_half);

        t += 0.5 * dt;
        const double mu = dt * K(t) * ((it == nt) ? 0.5 : 1.0);
        for (int i = 0; i < ndof; ++i)
        {
            p(i) -= dt * q_half(i);
            q(i) += dt * p_half(i);
            
            u(i) += mu * p(i);
            v(i) += mu * q(i);
        }
    }
}