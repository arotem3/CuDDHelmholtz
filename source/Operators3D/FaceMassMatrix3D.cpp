#include "Operators3D/FaceMassMatrix3D.hpp"

#include "forall.hpp"

using namespace cuddh;

static void init_face_mass(const TraceSpace3D &tr, const double *a, double *m)
{
    const Basis &basis = tr.h1_space().basis();
    const auto &quad = basis.quadrature();
    const auto &mesh = tr.h1_space().mesh().to_device();

    const int n_faces = tr.n_faces();
    const int n_basis = basis.size();

    auto w = quad.w(MemorySpace::DEVICE);
    auto x = quad.x(MemorySpace::DEVICE);

    auto I = tr.subspace_indices(MemorySpace::DEVICE);

    forall_2d(n_basis, n_basis, n_faces, [=] __device__(int f) mutable -> void {
        const int i = threadIdx.x;
        const int j = threadIdx.y;

        __shared__ QuadFace face;
        if (i == 0 && j == 0)
            face = mesh.face(f);

        const int idx = I(i, j, f);
        double a_val = (a) ? a[idx] : 1.0;
        a_val *= w(i) * w(j);

        const double2 r{x(i), x(j)};

        __syncthreads();

        a_val *= face.measure(r);
        atomicAdd(m + idx, a_val);
    });
}

FaceMassMatrix3D::FaceMassMatrix3D(const TraceSpace3D &tr) : Operator<double>(tr.size()), tr{tr}, m(tr.size())
{
    init_face_mass(tr, nullptr, m.device_write());
}

FaceMassMatrix3D::FaceMassMatrix3D(const double *a, const TraceSpace3D &tr)
    : Operator<double>(tr.size()), tr{tr}, m(tr.size())
{
    init_face_mass(tr, a, m.device_write());
}

void FaceMassMatrix3D::action(double c, const double *x, double *y) const
{
    const int n = tr.size();
    auto m = this->m.device_read();

    forall(n, [=] __device__(int i) -> void { y[i] += c * m[i] * x[i]; });
}

void FaceMassMatrix3D::action(const double *x, double *y) const
{
    const int n = tr.size();
    auto m = this->m.device_read();

    forall(n, [=] __device__(int i) -> void { y[i] = m[i] * x[i]; });
}
