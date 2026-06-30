#include "Operators2D/MassMatrix.hpp"

#include <vector>

#include "forall.hpp"

using namespace cuddh;

static inline thrust::device_vector<double> compute_mass_matrix(const H1Space2D &fem, const GridFunc2D<double> *_a)
{
    const int n_elem = fem.mesh().n_elem();
    const int n_basis = fem.basis().size();

    auto &q = fem.basis().quadrature();
    auto I = fem.global_indices(MemorySpace::DEVICE);

    auto w = q.w(MemorySpace::DEVICE);
    auto x = q.x(MemorySpace::DEVICE);

    auto d_mesh = fem.mesh().to_device();

    TensorWrapper<3, const double> a;
    if (_a)
        a = _a->read(MemorySpace::DEVICE);

    thrust::device_vector<double> _m(fem.size(), 0.0);
    double *d_m = thrust::raw_pointer_cast(_m.data());

    forall_2d(n_basis, n_basis, n_elem, [=] __device__(int el) {
        const int i = threadIdx.x;
        const int j = threadIdx.y;

        __shared__ QuadElement element;
        if (i == 0 && j == 0)
            element = d_mesh.element(el);
        __syncthreads();

        const int idx = I(i, j, el);
        double m = w(i) * w(j) * element.measure({x(i), x(j)});
        if (a)
            m *= a(i, j, el);

        atomicAdd(d_m + idx, m);
    });

    return _m;
}

cuddh::MassMatrix::MassMatrix(const H1Space2D &fem_, const GridFunc2D<double> &a)
    : Operator<double>(fem_.size()), fem(fem_)
{
    _m = compute_mass_matrix(fem_, &a);
}

cuddh::MassMatrix::MassMatrix(const H1Space2D &fem_) : Operator<double>(fem_.size()), fem(fem_)
{
    _m = compute_mass_matrix(fem_, nullptr);
}

void cuddh::MassMatrix::action(double c, const double *x, double *y) const
{
    auto m = to_device();

    forall(m.size(), [=] __device__(int i) -> void { y[i] += c * m(i) * x[i]; });
}

void cuddh::MassMatrix::action(const double *x, double *y) const
{
    auto m = to_device();

    forall(m.size(), [=] __device__(int i) -> void { y[i] = m(i) * x[i]; });
}

bool cuddh::MassMatrix::assemble(double c, SparseMatrix<double> &S) const
{
    std::vector<double> h_m(_m.size());
    thrust::copy(_m.begin(), _m.end(), h_m.begin());
    const int n = static_cast<int>(h_m.size());
    for (int i = 0; i < n; ++i)
        S.add_entry(i, i, c * h_m[i]);
    return true;
}

bool cuddh::MassMatrix::assemble(std::complex<double> c, SparseMatrix<double, true> &S) const
{
    std::vector<double> h_m(_m.size());
    thrust::copy(_m.begin(), _m.end(), h_m.begin());
    const int n = static_cast<int>(h_m.size());
    for (int i = 0; i < n; ++i)
        S.add_entry(i, i, c * h_m[i]);
    return true;
}
