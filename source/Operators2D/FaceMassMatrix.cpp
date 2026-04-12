#include "Operators2D/FaceMassMatrix.hpp"

using namespace cuddh;

static thrust::universal_vector<double> construct_face_mass(const TraceSpace2D &fs, const double *d_a)
{
    const int n_faces = fs.n_faces();
    const int n_basis = fs.h1_space().basis().size();

    auto &q = fs.h1_space().basis().quadrature();

    auto face_indices = fs.faces(MemorySpace::DEVICE);
    auto d_mesh = fs.h1_space().mesh().to_device();
    auto I = fs.subspace_indices(MemorySpace::DEVICE);
    auto K = fs.global_indices(MemorySpace::DEVICE);

    thrust::universal_vector<double> _w(n_basis);
    for (int i = 0; i < n_basis; ++i)
        _w[i] = q.w(i);
    auto w = reshape(_w, n_basis);

    thrust::universal_vector<double> _m(fs.h1_space().size(), 0.0);
    double *d_m = thrust::raw_pointer_cast(_m.data());

    forall_1d(n_basis, n_faces, [=] __device__(int f) mutable -> void {
        const int k = threadIdx.x;
        const int fs_idx = I(k, f);
        const int fem_idx = K(fs_idx);

        const double ds = d_mesh.edge(face_indices(f)).measure();
        double a = w(k) * ds;
        if (d_a)
            a *= d_a[fs_idx];

        atomicAdd(d_m + fem_idx, a);
    });

    return _m;
}

cuddh::FaceMassMatrix::FaceMassMatrix(const TraceSpace2D &fs, const double *d_a)
    : Operator<double>(fs.h1_space().size())
{
    _m = construct_face_mass(fs, d_a);
}

void cuddh::FaceMassMatrix::action(double c, const double *x, double *y) const
{
    auto m = to_device();

    forall(m.size(), [=] __device__(int i) -> void { y[i] += c * m(i) * x[i]; });
}

void cuddh::FaceMassMatrix::action(const double *x, double *y) const
{
    auto m = to_device();

    forall(m.size(), [=] __device__(int i) -> void { y[i] = m(i) * x[i]; });
}
