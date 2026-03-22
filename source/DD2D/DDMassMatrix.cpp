#include "DD2D/DDMassMatrix.hpp"

using namespace cuddh;

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

    forall_2d(n_basis * n_basis, mx_elem_per_dom, n_domains, [=] __device__(int subsp) mutable {
        const int s_nel = d_n_elems(subsp);

        const int i = threadIdx.x % n_basis;
        const int j = threadIdx.x / n_basis;
        const int el = threadIdx.y;

        if (el < s_nel)
        {
            const int g_el = d_elems(el, subsp);
            int l = sI(i, j, el, subsp);
            scalar_t val = static_cast<scalar_t>(w(i) * w(j) * detJ(i, j, g_el));
            atomicAdd(&M(l, subsp), val);
        }
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