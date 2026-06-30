#include "Operators2D/FaceMassMatrix.hpp"

#include <vector>

#include "FEM2D/TraceFunc2D.hpp"
#include "forall.hpp"

using namespace cuddh;

static thrust::device_vector<double> construct_face_mass(const TraceSpace2D &fs, const TraceFunc2D<double> *_a)
{
    const int n_faces = fs.n_faces();
    const int n_basis = fs.h1_space().basis().size();

    auto w = fs.h1_space().basis().quadrature().w(MemorySpace::DEVICE);

    auto face_indices = fs.faces(MemorySpace::DEVICE);
    auto d_mesh = fs.h1_space().mesh().to_device();
    auto I = fs.subspace_indices(MemorySpace::DEVICE);
    auto K = fs.global_indices(MemorySpace::DEVICE);

    MatrixWrapper<const double> a;
    if (_a)
        a = _a->read(MemorySpace::DEVICE);

    thrust::device_vector<double> _m(fs.h1_space().size(), 0.0);
    double *d_m = thrust::raw_pointer_cast(_m.data());

    forall_1d(n_basis, n_faces, [=] __device__(int f) {
        const int k = threadIdx.x;
        const int fs_idx = I(k, f);
        const int fem_idx = K(fs_idx);

        const double ds = d_mesh.edge(face_indices(f)).measure();
        double m = w(k) * ds;
        if (a)
            m *= a(k, f);

        atomicAdd(d_m + fem_idx, m);
    });

    return _m;
}

cuddh::FaceMassMatrix::FaceMassMatrix(const TraceSpace2D &fs, const GridFunc2D<double> &a)
    : Operator<double>(fs.h1_space().size())
{
    TraceFunc2D<double> tr_a = trace(fs, a);
    _m = construct_face_mass(fs, &tr_a);
}

cuddh::FaceMassMatrix::FaceMassMatrix(const TraceSpace2D &fs) : Operator<double>(fs.h1_space().size())
{
    _m = construct_face_mass(fs, nullptr);
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

bool cuddh::FaceMassMatrix::assemble(double c, SparseMatrix<double> &S) const
{
    std::vector<double> h_m(_m.size());
    thrust::copy(_m.begin(), _m.end(), h_m.begin());
    const int n = static_cast<int>(h_m.size());
    for (int i = 0; i < n; ++i)
        S.add_entry(i, i, c * h_m[i]);
    return true;
}

bool cuddh::FaceMassMatrix::assemble(std::complex<double> c, SparseMatrix<double, true> &S) const
{
    std::vector<double> h_m(_m.size());
    thrust::copy(_m.begin(), _m.end(), h_m.begin());
    const int n = static_cast<int>(h_m.size());
    for (int i = 0; i < n; ++i)
        S.add_entry(i, i, c * h_m[i]);
    return true;
}
