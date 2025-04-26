#include "Operators2D/WaveEquation.hpp"

using namespace cuddh;

static void init_mass(const H1Space2D &fem, const double *d_a, double *d_m)
{
    const int n_elem = fem.mesh().n_elem();
    const int n_basis = fem.basis().size();

    auto& q = fem.basis().quadrature();
    auto& metrics = fem.mesh().element_metrics(q);
    auto detJ = reshape(metrics.measures(MemorySpace::DEVICE), n_basis, n_basis, n_elem);
    auto I = fem.global_indices(MemorySpace::DEVICE);

    host_device_dvec _w(n_basis);
    double * h_w = _w.host_write();
    for (int i = 0; i < n_basis; ++i)
        h_w[i] = q.w(i);
    auto w = reshape(_w.device_read(), n_basis);

    forall_2d(n_basis, n_basis, n_elem, [=] __device__ (int el) mutable -> void
    {
        const int i = threadIdx.x;
        const int j = threadIdx.y;

        const int idx = I(i, j, el);
        double m = w(i) * w(j) * detJ(i, j, el);
        if (d_a)
            m *= d_a[idx];

        atomicAdd(d_m + idx, m);
    });
}

static void init_face_mass(const TraceSpace2D &fs, const double *d_a, double *d_m)
{
    const int n_faces = fs.n_faces();
    const int n_basis = fs.h1_space().basis().size();

    auto& q = fs.h1_space().basis().quadrature();
    auto& metrics = fs.metrics(q);
    auto detJ = reshape(metrics.measures(MemorySpace::DEVICE), n_basis, n_faces);
    
    auto I = fs.subspace_indices(MemorySpace::DEVICE);
    auto K = fs.global_indices(MemorySpace::DEVICE);

    host_device_dvec _w(n_basis);
    double * h_w = _w.host_write();
    for (int i = 0; i < n_basis; ++i)
        h_w[i] = q.w(i);
    auto w = reshape(_w.device_read(), n_basis);

    forall_1d(n_basis, n_faces, [=] __device__ (int f) mutable -> void
    {
        const int k = threadIdx.x;
        const int fs_idx = I(k, f);
        const int fem_idx = K(fs_idx);

        double a = w(k) * detJ(k, f);
        if (d_a)
            a *= d_a[fs_idx];

        atomicAdd(d_m + fem_idx, a);
    });
}

WaveEquation::WaveEquation(const double * a2x, const double * ax, const H1Space2D& fem_, const TraceSpace2D& fs_)
    : ndof{fem_.size()},
      fem{fem_},
      fs{fs_},
      S(fem_),
      M(ndof),
      H(ndof)
{
    init_mass(fem, a2x, M.device_write());
    init_face_mass(fs, ax, H.device_write());
}

void WaveEquation::step(double dt, const double * d_u0, double * d_u1, const double *a0, double *a1) const
{
    const double * p = d_u0;
    const double * q = d_u0 + ndof;

    double * p1 = d_u1;
    double * q1 = d_u1 + ndof;

    auto m = reshape(M.device_read(), ndof);
    auto h = reshape(H.device_read(), ndof);

    // p1 <- p + dt * q + 0.5 * dt^2 M \ a0
    forall(ndof, [=] __device__ (int i) -> void
    {
        p1[i] = p[i] + dt * q[i] + 0.5 * dt * dt * a0[i] / m[i];
    });

    // a1 <- a1 - S(p1)
    S.action(-1.0, p1, a1);

    // q1 <- (M + dt/2 * H) \ (M * q + dt/2 * a0 + dt/2 * a1)
    // a1 <- a1 - H * q1
    forall(ndof, [=] __device__ (int i) -> void
    {
        double r = m[i] * q[i] + 0.5 * dt * a0[i] + 0.5 * dt * a1[i];
        q1[i] = r / (m[i] + 0.5 * dt * h[i]);

        a1[i] -= h[i] * q1[i];
    });
}

void WaveEquation::initialize_acceleration(const double * d_u0, double * d_acc0) const
{
    const double * p = d_u0;
    const double * q = d_u0 + ndof;

    auto h = reshape(H.device_read(), ndof);

    // d_acc0 <- d_acc0 - H * q
    forall(ndof, [=] __device__ (int i) -> void
    {
        d_acc0[i] -= h[i] * q[i];
    });

    // d_acc0 <- d_acc0 - S(p)
    S.action(-1.0, p, d_acc0);
}
