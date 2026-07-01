#include "Operators3D/MassMatrix3D.hpp"

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/transform_reduce.h>
#include <thrust/tuple.h>

#include "FEM3D/GridFunc3D.hpp"
#include "forall.hpp"
#include "linalg.hpp"

using namespace cuddh;

static HostDeviceArray<double> init_mass(const H1Space3D &fem, const GridFunc3D<double> *a)
{
    const auto &quad = fem.basis().quadrature();
    const auto &mesh = fem.mesh().to_device();
    const int n_elem = fem.mesh().n_elem();
    const int n_basis = quad.size();

    auto I = fem.global_indices(MemorySpace::DEVICE);

    TensorWrapper<4, const double> A;
    if (a)
        A = a->read(MemorySpace::DEVICE);

    auto w = quad.w(MemorySpace::DEVICE);
    auto x = quad.x(MemorySpace::DEVICE);

    HostDeviceArray<double> M(fem.size());
    double *d_M = M.device_write();

    forall_3d(n_basis, n_basis, n_basis, n_elem, [=] __device__(int el) mutable -> void {
        const int i = threadIdx.x;
        const int j = threadIdx.y;
        const int k = threadIdx.z;
        const int idx = I(i, j, k, el);

        __shared__ HexElement elem;
        if (i == 0 && j == 0 && k == 0)
            elem = mesh.element(el);
        __syncthreads();

        const double3 r{x(i), x(j), x(k)};
        double m = w(i) * w(j) * w(k) * elem.measure(r);

        if (A)
            m *= A(i, j, k, el);

        atomicAdd(d_M + idx, m);
    });

    return M;
}

MassMatrix3D::MassMatrix3D(const H1Space3D &fem) : Operator<double>(fem.size()), fem{fem}
{
    _m = init_mass(fem, nullptr);
}

MassMatrix3D::MassMatrix3D(const H1Space3D &fem, const GridFunc3D<double> &a) : Operator<double>(fem.size()), fem{fem}
{
    _m = init_mass(fem, &a);
}

void MassMatrix3D::action(double c, const double *x, double *y) const
{
    const int n = fem.size();
    auto m = _m.device_read();

    forall(n, [=] __device__(int i) -> void { y[i] += c * m[i] * x[i]; });
}

void MassMatrix3D::action(const double *x, double *y) const
{
    const int n = fem.size();
    auto m = _m.device_read();

    forall(n, [=] __device__(int i) -> void { y[i] = m[i] * x[i]; });
}

InvMassMatrix3D MassMatrix3D::inv() const
{
    return InvMassMatrix3D(*this);
}

static void inv_mass(int n, double *d_m)
{
    forall(n, [=] __device__(int i) -> void { d_m[i] = 1.0 / d_m[i]; });
}

InvMassMatrix3D::InvMassMatrix3D(const H1Space3D &fem) : Operator<double>(fem.size()), fem{fem}
{
    _mi = init_mass(fem, nullptr);
    inv_mass(fem.size(), _mi.device_write());
}

InvMassMatrix3D::InvMassMatrix3D(const H1Space3D &fem, const GridFunc3D<double> &a)
    : Operator<double>(fem.size()), fem{fem}
{
    _mi = init_mass(fem, &a);
    inv_mass(fem.size(), _mi.device_write());
}

InvMassMatrix3D::InvMassMatrix3D(const MassMatrix3D &M) : Operator<double>(M.fem.size()), fem{M.fem}, _mi{M._m}
{
    inv_mass(fem.size(), _mi.device_write());
}

void InvMassMatrix3D::action(double c, const double *x, double *y) const
{
    const int n = fem.size();
    auto mi = _mi.device_read();

    forall(n, [=] __device__(int i) -> void { y[i] += c * mi[i] * x[i]; });
}

void InvMassMatrix3D::action(const double *x, double *y) const
{
    const int n = fem.size();
    auto mi = _mi.device_read();

    forall(n, [=] __device__(int i) -> void { y[i] = mi[i] * x[i]; });
}

namespace
{
    struct l2_dot_op
    {
        __device__ double operator()(thrust::tuple<double, double, double> t) const
        {
            const auto [m, a, b] = t;
            return a * m * b;
        }
    };

    struct l2_dist_op
    {
        __device__ double operator()(thrust::tuple<double, double, double> t) const
        {
            const auto [m, a, b] = t;
            double e = a - b;
            return e * m * e;
        }
    };
} // anonymous namespace

double cuddh::l2_dot(const MassMatrix3D &M, const double *x, const double *y)
{
    const int n = M.fem.size();
    auto m = thrust::device_pointer_cast(M._m.device_read());
    auto px = thrust::device_pointer_cast(x);
    auto py = thrust::device_pointer_cast(y);

    auto begin = thrust::make_zip_iterator(thrust::make_tuple(m, px, py));
    auto end = thrust::make_zip_iterator(thrust::make_tuple(m + n, px + n, py + n));

    return thrust::transform_reduce(thrust::device, begin, end, ::l2_dot_op{}, 0.0, thrust::plus<double>());
}

double cuddh::l2_dist(const MassMatrix3D &M, const double *x, const double *y)
{
    const int n = M.fem.size();
    auto m = thrust::device_pointer_cast(M._m.device_read());
    auto px = thrust::device_pointer_cast(x);
    auto py = thrust::device_pointer_cast(y);

    auto begin = thrust::make_zip_iterator(thrust::make_tuple(m, px, py));
    auto end = thrust::make_zip_iterator(thrust::make_tuple(m + n, px + n, py + n));

    return std::sqrt(thrust::transform_reduce(thrust::device, begin, end, ::l2_dist_op{}, 0.0, thrust::plus<double>()));
}

bool cuddh::MassMatrix3D::assemble(double c, SparseMatrix<double> &S) const
{
    const int n = fem.size();
    const double *m = _m.host_read();
    for (int i = 0; i < n; ++i)
        S.set_value(i, i, c * m[i]);
    return true;
}

bool cuddh::MassMatrix3D::assemble(std::complex<double> c, SparseMatrix<double, true> &S) const
{
    const int n = fem.size();
    const double *m = _m.host_read();
    for (int i = 0; i < n; ++i)
        S.set_value(i, i, c * m[i]);
    return true;
}
