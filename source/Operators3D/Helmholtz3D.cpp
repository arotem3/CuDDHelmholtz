#include "Operators3D/Helmholtz3D.hpp"

using namespace cuddh;

static void init_face_mass(const TraceSpace3D &tr, const double *d_a, double *d_m)
{
    const int n_faces = tr.n_faces();
    const int n_basis = tr.h1_space().basis().size();

    auto &quad = tr.h1_space().basis().quadrature();

    host_device_dvec _w(n_basis);
    double *h_w = _w.host_write();
    for (int i = 0; i < n_basis; ++i)
        h_w[i] = quad.w(i);
    auto w = reshape(_w.device_read(), n_basis);

    host_device_dvec _x(n_basis);
    double *h_x = _x.host_write();
    for (int i = 0; i < n_basis; ++i)
        h_x[i] = quad.x(i);
    auto x = reshape(_x.device_read(), n_basis);

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

Helmholtz3D::Helmholtz3D(double omega_, const double *a2x, const double *ax, const H1Space3D &fem,
                         const TraceSpace3D &tr)
    : Operator<double>(2 * fem.size()), omega{omega_}, S(fem), M(a2x, fem), H(fem.size())
{
    init_face_mass(tr, ax, H.device_write());
}

void Helmholtz3D::action(const double *x, double *y) const
{
    const int n = this->ndof() / 2;

    const double *u = x;
    const double *v = x + n;

    double *Au = y;
    double *Av = y + n;

    S.action(u, Au);
    S.action(v, Av);

    double omega = this->omega;
    auto m = diagonal_mass(M, MemorySpace::DEVICE);
    auto h = reshape(H.device_read(), n);

    forall(n, [=] __device__(int i) -> void {
        const double mi = m[i];
        const double hi = h[i];

        const double U = u[i], V = v[i];

        Au[i] = Au[i] - omega * omega * mi * U + omega * hi * V;
        Av[i] = -Av[i] + omega * omega * mi * V + omega * hi * U;
    });
}
