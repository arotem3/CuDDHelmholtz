#include "Operators3D/WaveEquation3D.hpp"

#include "forall.hpp"
#include "linalg.hpp"

using namespace cuddh;

static GridFunc3D<double> square(const GridFunc3D<double> &a)
{
    return a.transform([] __device__(double x) -> double { return x * x; });
}

WaveEquation3D::WaveEquation3D(const H1Space3D &fem, const TraceSpace3D &tr)
    : ndof(fem.size()), fem{fem}, tr{tr}, S(fem), M(fem), H(tr)
{}

WaveEquation3D::WaveEquation3D(const H1Space3D &fem, const TraceSpace3D &tr, const GridFunc3D<double> &a)
    : ndof(fem.size()), fem{fem}, tr{tr}, S(fem), M(fem, square(a)), H(tr, a)
{}

void WaveEquation3D::step(double dt, const double *d_u0, double *d_u1, const double *a0, double *a1) const
{
    const double *p = d_u0;
    const double *q = d_u0 + ndof;

    double *p1 = d_u1;
    double *q1 = d_u1 + ndof;

    auto m = M.to_device();
    auto h = H.to_device();

    // p1 <- p + dt * q + 0.5 * dt^2 M \ a0
    forall(ndof, [=] __device__(int i) -> void { p1[i] = p[i] + dt * q[i] + 0.5 * dt * dt * a0[i] / m(i); });

    // a1 <- a1 - S(p1)
    S.action(-1.0, p1, a1);

    // q1 <- (M + dt/2 * H) \ (M * q + dt/2 * a0 + dt/2 * a1)
    // a1 <- a1 - H * q1
    forall(ndof, [=] __device__(int i) -> void {
        double r = m(i) * q[i] + 0.5 * dt * a0[i] + 0.5 * dt * a1[i];
        q1[i] = r / (m(i) + 0.5 * dt * h(i));

        a1[i] -= h(i) * q1[i];
    });
}

void WaveEquation3D::initialize_acceleration(const double *d_u0, double *d_acc0) const
{
    const double *p = d_u0;
    const double *q = d_u0 + ndof;

    auto h = H.to_device();

    forall(ndof, [=] __device__(int i) -> void { d_acc0[i] -= h(i) * q[i]; });

    S.action(-1.0, p, d_acc0);
}
