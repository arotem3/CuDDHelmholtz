#include "Operators3D/MassMatrix3D.hpp"

using namespace cuddh;

static void init_mass(const H1Space3D &fem, const double *d_a, double *d_M)
{
    const auto &quad = fem.basis().quadrature();
    const auto &mesh = fem.mesh().to_device();
    
    const int n_elem = fem.mesh().n_elem();
    const int n_basis = quad.size();
    const int ndof = fem.size();
    
    auto I = reshape(fem.global_indices(MemorySpace::DEVICE), n_basis, n_basis, n_basis, n_elem);

    host_device_dvec _w(n_basis);
    double * h_w = _w.host_write();
    for (int i = 0; i < n_basis; ++i)
        h_w[i] = quad.w(i);
    auto w = reshape(_w.device_read(), n_basis);

    host_device_dvec _x(n_basis);
    double * h_x = _x.host_write();
    for (int i = 0; i < n_basis; ++i)
        h_x[i] = quad.x(i);
    auto x = reshape(_x.device_read(), n_basis);

    forall_3d(n_basis, n_basis, n_basis, n_elem, [=] __device__ (int el) mutable -> void
    {
        const int i = threadIdx.x;
        const int j = threadIdx.y;
        const int k = threadIdx.z;

        const int idx = I(i, j, k, el);

        __shared__ HexElement elem;

        if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0)
            elem = mesh.element(el);
        __syncthreads();

        const double3 r{x(i), x(j), x(k)};
        const double detJ = elem.measure(r);

        double m = w(i) * w(j) * w(k) * detJ;
        m *= (d_a) ? d_a[idx] : 1.0;

        atomicAdd(d_M + idx, m);
    });
}

MassMatrix3D::MassMatrix3D(const H1Space3D &fem)
    : fem{fem},
      _m(fem.size())
{
    init_mass(fem, nullptr, _m.device_write());
}

MassMatrix3D::MassMatrix3D(const double *d_a, const H1Space3D &fem)
    : fem{fem},
      _m(fem.size())
{
    init_mass(fem, d_a, _m.device_write());
}

void MassMatrix3D::action(double c, const double *x, double *y) const
{
    const int n = fem.size();
    auto m = _m.device_read();

    forall(n, [=] __device__ (int i) -> void
    {
        y[i] += c * m[i] * x[i];
    });
}

void MassMatrix3D::action(const double *x, double *y) const
{
    const int n = fem.size();
    auto m = _m.device_read();

    forall(n, [=] __device__ (int i) -> void
    {
        y[i] = m[i] * x[i];
    });
}

InvMassMatrix3D MassMatrix3D::inv() const
{
    return InvMassMatrix3D(*this);
}

static void inv_mass(int n, const double * __restrict__ d_m, double * __restrict__ d_mi)
{
    forall(n, [=] __device__ (int i) -> void
    {
        d_mi[i] = 1.0 / d_m[i];
    });
}

// inplace
static void inv_mass(int n, double * d_m)
{
    forall(n, [=] __device__ (int i) -> void
    {
        d_m[i] = 1.0 / d_m[i];
    });
}

InvMassMatrix3D::InvMassMatrix3D(const H1Space3D &fem)
    : fem{fem},
      _mi(fem.size())
{
    init_mass(fem, nullptr, _mi.device_write());
    inv_mass(fem.size(), _mi.device_write());
}

InvMassMatrix3D::InvMassMatrix3D(const double *d_a, const H1Space3D &fem)
    : fem{fem},
      _mi(fem.size())
{
    init_mass(fem, d_a, _mi.device_write());
    inv_mass(fem.size(), _mi.device_write());
}

InvMassMatrix3D::InvMassMatrix3D(const MassMatrix3D &M)
    : fem{M.fem},
      _mi(fem.size())
{
    const double *d_m = M._m.device_read();
    double *d_mi = _mi.device_write();
    
    inv_mass(fem.size(), d_m, d_mi);
}

void InvMassMatrix3D::action(double c, const double *x, double *y) const
{
    const int n = fem.size();
    auto mi = _mi.device_read();

    forall(n, [=] __device__ (int i) -> void
    {
        y[i] += c * mi[i] * x[i];
    });
}

void InvMassMatrix3D::action(const double *x, double *y) const
{
    const int n = fem.size();
    auto mi = _mi.device_read();

    forall(n, [=] __device__ (int i) -> void
    {
        y[i] = mi[i] * x[i];
    });
}
