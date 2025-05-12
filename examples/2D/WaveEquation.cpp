/**
 * @file WaveEquation.cpp
 * @brief Example driver for solving the wave equation in 2D.
 * 
 * @details This file is a driver for solving the wave equation:
 * 
 *     alpha(x)^2 u_{tt} - div(grad u) == f cos(omega*t)    in  D := [-1, 1]x[-1, 1]
 *     alpha(x) u_t + dU/dn == 0                            on boundary of D
 * 
 * Here alpha(x) is a variable coefficient, and omega is the frequency.
 * We assume f and u are real valued.
 * 
 * The WaveEquation class implements the finite element discretization of this equation.
 * 
 * The program will write the collocation points to `solution/xy.0000` in binary
 * format. The solution is written to `solution/wave{:05d}.0000` in binary where
 * {:05d} is the time step.
 */

#include "cuddh.hpp"
#include "examples.hpp"
#include <format>

using namespace cuddh;

__device__ static double f(const double X[2], double omega)
{
    const double x = X[0], y = X[1];
    double s = omega * omega;
    
    double r = (x+0.5)*(x+0.5) + y * y;
    double F = s / M_PI * std::exp(-s * r);
    
    r = (x-0.5)*(x-0.5) + (y+0.5)*(y+0.5);
    F += s / M_PI * std::exp(-s * r);
    return F;
}

__device__ static double alpha(const double X[2])
{
    const double r = X[0] * X[0] + X[1] * X[1];

    if (r < 0.0625)
        return 0.2;
    else
        return 1.0;
}

constexpr double maxvel = 5.0; // maximum reciprocal of alpha

int main()
{
    const int deg = 2; // polynomial degree
    const int nx = 128; // number of elements in x direction
    const double omega = 2 * M_PI * nx / 10; // time-harmonic frequency
    const double T = 2.0; // total time

    // Create a uniform rectangular mesh
    Mesh2D mesh = Mesh2D::uniform_rect(nx, -1.0, 1.0, nx, -1.0, 1.0);

    // Construct 1D basis functions
    Basis basis(deg + 1);

    // Combine the mesh and basis functions in H1Space2D to define the total global degrees of freedom
    H1Space2D fem(mesh, basis);

    const int ndof = fem.size(); // number of degrees of freedom
    const int N = 2 * ndof; // total degrees of freedom in [u, v] (v = u_t)

    ivec boundary_faces = mesh.boundary_edges(); // identify boundary faces
    TraceSpace2D fs(fem, boundary_faces.size(), boundary_faces); // define trace space

    const int fdof = fs.size(); // number of degrees of freedom in trace space

    host_device_dvec U(N); // solution vector
    host_device_dvec U1(N); // updated solution vector
    host_device_dvec b(N); // forcing term
    host_device_dvec a2(ndof); /// variable coefficient on collocation points
    host_device_dvec a(fdof); // variable coefficient on trace space
    host_device_dvec acc(ndof); // acceleration vector
    host_device_dvec acc1(ndof); // updated acceleration vector

    double *d_b = b.device_write(); // device pointer to right-hand side vector
    double *d_a2 = a2.device_write(); // device pointer to variable coefficient
    double *d_a = a.device_write(); // device pointer to variable coefficient on trace space
    
    LinearFunctional l(fem);
    DiagInvMassMatrix mi(fem);
    l.action([=] __device__(const double X[2]) -> double { return f(X, omega); }, d_b); // compute right-hand side
    
    l.action([=] __device__(const double X[2]) -> double { double ax = alpha(X); return ax*ax; }, d_a2); // compute variable coefficient
    mi.action(d_a2, d_a2); // project the coefficient onto the basis

    FaceLinearFunctional fl(fs);
    DiagInvFaceMassMatrix fmi(fs);

    fl.action([=] __device__(const double X[2]) -> double { return alpha(X); }, d_a); // compute variable coefficient on trace space
    fmi.action(d_a, d_a); // project the coefficient onto the trace space

    // initialize wave equation operator
    double dt = 2.0 * mesh.min_h() / (maxvel * (deg+1) * (deg+1)); // CFL condition
    int nt = T / dt;
    dt = T / nt;

    WaveEquation W(d_a2, d_a, fem, fs);

    std::cout << "Solving the wave equation...\n"
              << "\tomega = " << omega << "\n"
              << "\t#elements = " << mesh.n_elem() << "\n"
              << "\tpolynomial degree = " << deg << "\n"
              << "\t#dof = " << 2 * ndof << "\n"
              << "\ttotal time = " << T << "\n"
              << "\tdt = " << dt << "\n"
              << "\t#time steps = " << nt << std::endl;

    copy(ndof, d_b, acc.device_write()); // acc = b * cos(0) = b
    W.initialize_acceleration(U.device_read_write(), acc.device_write()); // set the acceleration vector to the acceleration at t = 0

    // save the initial condition and grid to file
    auto xy = fem.physical_coordinates(MemorySpace::HOST);
    to_file("solution/xy.0000", 2*ndof, xy.data());

    to_file(std::format("solution/wave{:05d}.0000", 0), ndof, U.host_read());

    double t = 0.0;
    for (int it = 1; it <= nt; ++it)
    {
        // set the forcing
        axpby(ndof, std::cos(omega * (t+dt)), d_b, 0.0, acc1.device_write()); // acc1 = b * cos(omega * (t+dt))
        W.step(dt, U.device_write(), U1.device_write(), acc.device_write(), acc1.device_write()); // step the wave equation

        std::swap(U, U1); // swap the solution vectors
        std::swap(acc, acc1); // swap the acceleration vectors
        t += dt;

        std::cout << std::fixed << std::setprecision(4)
                  << "t = " << std::setw(10) << t << "\r" << std::flush;

        // save the solution to a file
        to_file(std::format("solution/wave{:05d}.0000", it), ndof, U.host_read());
    }
    std::cout << std::endl;

    return 0;
}