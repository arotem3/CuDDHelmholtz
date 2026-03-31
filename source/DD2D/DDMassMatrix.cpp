#include "DD2D/DDMassMatrix.hpp"

using namespace cuddh;

static constexpr __device__ int4 get_indices(int t, int4 dims)
{
    int4 i;

    int bw = dims.x * dims.y * dims.z;
    i.w = t / bw;
    t = t % bw;

    int bz = dims.x * dims.y;
    i.z = t / bz;
    t = t % bz;

    i.y = t / dims.x;
    i.x = t % dims.x;

    return i;
}

template <typename scalar_t>
static void mass(scalar_t *d_m, const H1Space2D &fem, const EnsembleSpace &efem)
{
    const Mesh2D &mesh = fem.mesh();
    const Basis &basis = fem.basis();
    const QuadratureRule &q = basis.quadrature();

    const int n_basis = basis.size();
    const int mx_elem_per_dom = efem.max_n_elem();
    const int n_domains = efem.size();
    const int mx_dofs = efem.max_size();

    host_device_dvec _w(n_basis);
    double *h_w = _w.host_write();
    for (int i = 0; i < n_basis; ++i)
        h_w[i] = q.w(i);
    auto w = reshape(_w.device_read(), n_basis);

    auto d_n_elems = efem.n_elems(MemorySpace::DEVICE);
    auto d_elems = efem.elements(MemorySpace::DEVICE);
    auto sI = efem.subspace_indices(MemorySpace::DEVICE);

    const double *d_detJ = mesh.element_metrics(q).measures(MemorySpace::DEVICE);
    auto detJ = reshape(d_detJ, n_basis, n_basis, mesh.n_elem());

    auto M = reshape(d_m, mx_dofs, n_domains);

    forall(n_basis * n_basis * mx_elem_per_dom * n_domains, [=] __device__(int tid) mutable {
        const auto [i, j, el, subsp] = get_indices(tid, {n_basis, n_basis, mx_elem_per_dom, n_domains});

        if (el >= d_n_elems(subsp))
            return;

        const int g_el = d_elems(el, subsp);
        int l = sI(i, j, el, subsp);
        scalar_t val = w(i) * w(j) * detJ(i, j, g_el);
        atomicAdd(&M(l, subsp), val);
    });
}

template <typename scalar_t>
DDMassMatrix<scalar_t>::DDMassMatrix(const H1Space2D &fem, const EnsembleSpace &efem)
    : mx_dofs(efem.max_size()), n_domains(efem.size()), m(mx_dofs * n_domains)
{
    mass<scalar_t>(m.device_write(), fem, efem);
}

namespace cuddh
{
    template class DDMassMatrix<float>;
    template class DDMassMatrix<double>;
} // namespace cuddh