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

    host_device_dvec _q_pts(n_basis);
    double *h_q = _q_pts.host_write();
    for (int i = 0; i < n_basis; ++i)
        h_q[i] = q.x(i);
    auto q_pts = reshape(_q_pts.device_read(), n_basis);

    auto d_mesh = mesh.to_device();

    auto d_n_elems = efem.n_elems(MemorySpace::DEVICE);
    auto d_elems = efem.elements(MemorySpace::DEVICE);
    auto sI = efem.subspace_indices(MemorySpace::DEVICE);

    auto M = reshape(d_m, mx_dofs, n_domains);

    forall_2d(n_basis, n_basis, mx_elem_per_dom * n_domains, [=] __device__(int index) mutable {
        const auto [i, j, _] = threadIdx;
        const int el = index % mx_elem_per_dom;
        const int subsp = index / mx_elem_per_dom;

        if (el >= d_n_elems(subsp))
            return;

        __shared__ QuadElement element;
        if (i == 0 && j == 0)
            element = d_mesh.element(d_elems(el, subsp));
        __syncthreads();

        scalar_t val = w(i) * w(j) * element.measure({q_pts(i), q_pts(j)});

        int l = sI(i, j, el, subsp);
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