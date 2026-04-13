#include "Operators3D/WaveEquation3D.hpp"

#include "forall.hpp"
#include "linalg.hpp"

using namespace cuddh;

static void init_face_mass(const TraceSpace3D &tr, const double *d_a, double *d_m)
{
    const int n_faces = tr.n_faces();
    const int n_basis = tr.h1_space().basis().size();

    auto &quad = tr.h1_space().basis().quadrature();

    auto w = quad.w(MemorySpace::DEVICE);
    auto x = quad.x(MemorySpace::DEVICE);

    DeviceMesh3D mesh = tr.h1_space().mesh().to_device();

    auto I = tr.subspace_indices(MemorySpace::DEVICE);
    auto J = tr.global_indices(MemorySpace::DEVICE);

    forall_2d(n_basis, n_basis, n_faces, [=] __device__(int f) mutable -> void {
        const int tr_idx = I(threadIdx.x, threadIdx.y, f);
        const int fem_idx = J(tr_idx);

        __shared__ QuadFace face;
        if (threadIdx.x == 0 && threadIdx.y == 0)
            face = mesh.face(f);

        double value = w(threadIdx.x) * w(threadIdx.y);
        if (d_a)
            value *= d_a[tr_idx];

        __syncthreads();

        value *= face.measure(double2{x(threadIdx.x), x(threadIdx.y)});

        atomicAdd(d_m + fem_idx, value);
    });
}

WaveEquation3D::WaveEquation3D(const double *a2x, const double *ax, const H1Space3D &fem, const TraceSpace3D &tr)
    : ndof(fem.size()), fem{fem}, tr{tr}, S(fem), M(a2x, fem), H(ndof)
{
    init_face_mass(tr, ax, H.device_write());
}

void WaveEquation3D::step(double dt, const double *d_u0, double *d_u1, const double *a0, double *a1) const
{
    const double *p = d_u0;
    const double *q = d_u0 + ndof;

    double *p1 = d_u1;
    double *q1 = d_u1 + ndof;

    auto m = diagonal_mass(M, MemorySpace::DEVICE);
    auto h = reshape(H.device_read(), ndof);

    // p1 <- p + dt * q + 0.5 * dt^2 M \ a0
    forall(ndof, [=] __device__(int i) -> void { p1[i] = p[i] + dt * q[i] + 0.5 * dt * dt * a0[i] / m[i]; });

    // a1 <- a1 - S(p1)
    S.action(-1.0, p1, a1);

    // q1 <- (M + dt/2 * H) \ (M * q + dt/2 * a0 + dt/2 * a1)
    // a1 <- a1 - H * q1
    forall(ndof, [=] __device__(int i) -> void {
        double r = m[i] * q[i] + 0.5 * dt * a0[i] + 0.5 * dt * a1[i];
        q1[i] = r / (m[i] + 0.5 * dt * h[i]);

        a1[i] -= h[i] * q1[i];
    });
}

void WaveEquation3D::initialize_acceleration(const double *d_u0, double *d_acc0) const
{
    const double *p = d_u0;
    const double *q = d_u0 + ndof;

    auto h = reshape(H.device_read(), ndof);

    forall(ndof, [=] __device__(int i) -> void { d_acc0[i] -= h[i] * q[i]; });

    S.action(-1.0, p, d_acc0);
}
