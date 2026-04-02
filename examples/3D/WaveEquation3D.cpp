/**
 * @file WaveEquation.cpp
 * @brief Example driver for solving the wave equation in 3D.
 * 
 * @details This file is a driver for solving the wave equation:
 * 
 *     alpha(x)^2 u_{tt} - div(grad u) == f cos(omega*t)    in  D := [-1, 1]^3
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

__device__ static double f(double3 x)
{
    double s = 10;
    double r = (x.x-0.5)*(x.x-0.5) + x.y * x.y + x.z * x.z;
    return std::pow(s * s / M_PI, 1.5) * std::exp(-s*s * r);
}

__device__ static double alpha(double3 x)
{
    const double r = x.x * x.x + x.y * x.y + x.z * x.z;

    if (r < 0.0625)
        return 0.2;
    else
        return 1.0;
}

constexpr double maxvel = 5.0; // maximum reciprocal of alpha

int main()
{
    const int deg = 2; // polynomial degree
    const int nx = 64; // number of elements in x direction
    const double omega = 2 * M_PI * nx / 10; // time-harmonic frequency
    const double T = 4.0; // total time

    // Create a uniform rectangular mesh
    Mesh3D mesh = Mesh3D::uniform_cube(nx, -1.0, 1.0, nx, -1.0, 1.0, nx, -1.0, 1.0);

    // Construct 1D basis functions
    Basis basis(deg + 1);

    // Combine the mesh and basis functions in H1Space3D to define the total global degrees of freedom
    H1Space3D fem(mesh, basis);

    const int ndof = fem.size(); // number of degrees of freedom
    const int N = 2 * ndof; // total degrees of freedom in [u, v] (v = u_t)

    auto boundary_faces = mesh.get_boundary_faces(); // identify boundary faces
    TraceSpace3D fs(fem, boundary_faces.size(), boundary_faces); // define trace space

    const int fdof = fs.size(); // number of degrees of freedom in trace space

    host_device_dvec U(N); // solution vector
    host_device_dvec U1(N); // updated solution vector
    host_device_dvec b(N); // forcing term
    host_device_dvec a2(ndof); // variable coefficient on collocation points
    host_device_dvec a(fdof); // variable coefficient on trace space
    host_device_dvec acc(ndof); // acceleration vector
    host_device_dvec acc1(ndof); // updated acceleration vector

    double *d_b = b.device_write(); // device pointer to right-hand side vector
    double *d_a2 = a2.device_write(); // device pointer to variable coefficient
    double *d_a = a.device_write(); // device pointer to variable coefficient on trace space

    // compute b = (f, phi)
    MassMatrix3D M(fem);
    l2_project(M, [=] __device__(double3 x) -> double { return f(x); }, d_b); // compute right-hand side

    to_file("solution/b.0000", N, b.host_read());

    // compute variable coefficient
    gridfunc(fem, [=] __device__(double3 x) -> double { double ax = alpha(x); return ax*ax; }, d_a2); // compute variable coefficient

    // compute trace of the variable coefficient
    trace(fs, [=] __device__(double3 x) -> double { return alpha(x); }, d_a);

    // initialize wave equation operator
    double dt = 2.0 / ((nx - 1) * maxvel * (deg + 1) * (deg + 1));
    int nt = T / dt;
    dt = T / nt;

    WaveEquation3D W(d_a2, d_a, fem, fs);

    std::cout << "Solving the wave equation...\n"
              << "\tomega = " << omega << "\n"
              << "\t#elements = " << mesh.n_elem() << "\n"
              << "\tpolynomial degree = " << deg << "\n"
              << "\t#dof = " << 2 * ndof << "\n"
              << "\ttotal time = " << T << "\n"
              << "\tdt = " << dt << "\n"
              << "\t#time steps = " << nt << std::endl;

    dla::copy(ndof, d_b, acc.device_write());
    W.initialize_acceleration(U.device_read(), acc.device_write());

    // save the initial condition and grid to file
    auto xyz = fem.physical_coordinates(MemorySpace::HOST);
    to_file("solution/coo.0000", ndof, xyz.data());

    to_file(std::format("solution/wave{:05d}.0000", 0), N, U.host_read());

    double t = 0.0;
    for (int it = 1; it <= nt; ++it)
    {
        //set the forcing:
        dla::axpby(ndof, std::cos(omega * (t + dt)), d_b, 0.0, acc1.device_write());

        W.step(dt, U.device_read(), U1.device_write(), acc.device_read(), acc1.device_write());

        std::swap(U, U1);
        std::swap(acc, acc1);
        t += dt;

        std::cout << std::fixed << std::setprecision(4)
                  << "t = " << std::setw(8) << t << " / " << T << "\r" << std::flush;

        if (it % 100 == 0)
            to_file(std::format("solution/wave{:05d}.0000", it/100), N, U.host_read());
    }
    std::cout << std::endl;

    return 0;
}