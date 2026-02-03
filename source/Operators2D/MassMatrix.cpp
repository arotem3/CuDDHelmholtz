#include "Operators2D/MassMatrix.hpp"

using namespace cuddh;

static inline thrust::universal_vector<double> compute_mass_matrix(const H1Space2D &fem, const double *a)
{
    const int n_elem = fem.mesh().n_elem();
    const int n_basis = fem.basis().size();

    auto &q = fem.basis().quadrature();
    auto &metrics = fem.mesh().element_metrics(q);
    auto detJ = reshape(metrics.measures(MemorySpace::DEVICE), n_basis, n_basis, n_elem);
    auto I = fem.global_indices(MemorySpace::DEVICE);

    thrust::universal_vector<double> _w(n_basis);
    for (int i = 0; i < n_basis; ++i)
        _w[i] = q.w(i);
    auto w = reshape(_w, n_basis);

    thrust::universal_vector<double> _m(fem.size(), 0.0);
    double *d_m = thrust::raw_pointer_cast(_m.data());

    forall_2d(n_basis, n_basis, n_elem, [=] __device__(int el) mutable -> void {
        const int i = threadIdx.x;
        const int j = threadIdx.y;

        const int idx = I(i, j, el);
        double m = w(i) * w(j) * detJ(i, j, el);
        if (a)
            m *= a[idx];

        atomicAdd(d_m + idx, m);
    });

    return _m;
}

cuddh::MassMatrix::MassMatrix(const H1Space2D &fem_, const double *a_) : fem(fem_)
{
    _m = compute_mass_matrix(fem_, a_);
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
