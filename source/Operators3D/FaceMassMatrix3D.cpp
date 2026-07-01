#include "Operators3D/FaceMassMatrix3D.hpp"

#include "FEM3D/TraceFunc3D.hpp"
#include "forall.hpp"
#include "linalg.hpp"

using namespace cuddh;

static HostDeviceArray<double> init_face_mass(const TraceSpace3D &tr, const TraceFunc3D<double> *a)
{
    const int n_faces = tr.n_faces();
    const int n_basis = tr.h1_space().basis().size();

    const auto &quad = tr.h1_space().basis().quadrature();
    const auto w = quad.w(MemorySpace::DEVICE);
    const auto x = quad.x(MemorySpace::DEVICE);
    const auto mesh = tr.h1_space().mesh().to_device();

    const auto faces = tr.faces(MemorySpace::DEVICE);
    const auto I = tr.subspace_indices(MemorySpace::DEVICE);
    const auto J = tr.global_indices(MemorySpace::DEVICE);

    TensorWrapper<3, const double> A;
    if (a)
        A = a->read(MemorySpace::DEVICE);

    HostDeviceArray<double> m(tr.h1_space().size());
    auto d_m = m.device_write();

    forall_2d(n_basis, n_basis, n_faces, [=] __device__(int f) mutable {
        const auto [i, j, _] = threadIdx;

        __shared__ QuadFace face;
        if (i == 0 && j == 0)
            face = mesh.face(faces[f]);
        __syncthreads();

        const int fem_idx = J(I(i, j, f));
        double val = w(i) * w(j) * face.measure(double2{x(i), x(j)});

        if (A)
            val *= A(i, j, f);

        atomicAdd(d_m + fem_idx, val);
    });

    return m;
}

FaceMassMatrix3D::FaceMassMatrix3D(const TraceSpace3D &tr) : Operator<double>(tr.h1_space().size()), _fem{tr.h1_space()}
{
    _m = init_face_mass(tr, nullptr);
}

FaceMassMatrix3D::FaceMassMatrix3D(const TraceSpace3D &tr, const GridFunc3D<double> &a)
    : Operator<double>(tr.h1_space().size()), _fem{tr.h1_space()}
{
    TraceFunc3D<double> tf = trace(tr, a);
    _m = init_face_mass(tr, &tf);
}

void FaceMassMatrix3D::action(double c, const double *x, double *y) const
{
    auto m = to_device();
    forall(m.size(), [=] __device__(int i) -> void { y[i] += c * m(i) * x[i]; });
}

void FaceMassMatrix3D::action(const double *x, double *y) const
{
    auto m = to_device();
    forall(m.size(), [=] __device__(int i) -> void { y[i] = m(i) * x[i]; });
}

bool cuddh::FaceMassMatrix3D::assemble(double c, SparseMatrix<double> &S) const
{
    const int n = _fem.size();
    const double *m = _m.host_read();
    for (int i = 0; i < n; ++i)
        S.set_value(i, i, c * m[i]);
    return true;
}

bool cuddh::FaceMassMatrix3D::assemble(std::complex<double> c, SparseMatrix<double, true> &S) const
{
    const int n = _fem.size();
    const double *m = _m.host_read();
    for (int i = 0; i < n; ++i)
        S.set_value(i, i, c * m[i]);
    return true;
}
