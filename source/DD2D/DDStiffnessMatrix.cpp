#include "DD2D/DDStiffnessMatrix.hpp"

using namespace cuddh;

template <typename scalar_t>
static void make_diffmat(scalar_t *h_D, const Basis &basis)
{
    const int n_basis = basis.size();
    dmat D(n_basis, n_basis);
    basis.deriv(n_basis, basis.quadrature().x(), D);
    for (int i = 0; i < n_basis * n_basis; ++i)
        h_D[i] = scalar_t(D[i]);
}

template <typename scalar_t>
static void geom_factors(SmallSymmetricMatrix<scalar_t, 2> *d_G, const H1Space2D &fem, const EnsembleSpace &efem)
{
    const Mesh2D &mesh = fem.mesh();
    const Basis &basis = fem.basis();
    const QuadratureRule &q = basis.quadrature();

    const int n_basis = basis.size();
    const int n_domains = efem.size();
    const int mx_elem = efem.max_n_elem();

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

    auto n_elems = efem.n_elems(MemorySpace::DEVICE);
    auto elems = efem.elements(MemorySpace::DEVICE);

    auto G = reshape(d_G, n_basis, n_basis, mx_elem, n_domains);

    forall_2d(n_basis, n_basis, mx_elem * n_domains, [=] __device__(int index) mutable -> void {
        const auto [i, j, _] = threadIdx;
        const int el = index % mx_elem;
        const int subsp = index / mx_elem;

        if (el >= n_elems[subsp])
            return;

        __shared__ QuadElement element;
        if (i == 0 && j == 0)
            element = d_mesh.element(elems(el, subsp));
        __syncthreads();

        const double2x2 J = element.jacobian({q_pts(i), q_pts(j)});
        const double W = w(i) * w(j) / det(J);

        SmallSymmetricMatrix<scalar_t, 2> gij;
        gij(0, 0) = W * (J(1, 1) * J(1, 1) + J(0, 1) * J(0, 1));
        gij(1, 0) = -W * (J(1, 0) * J(1, 1) + J(0, 0) * J(0, 1));
        gij(1, 1) = W * (J(1, 0) * J(1, 0) + J(0, 0) * J(0, 0));

        G(i, j, el, subsp) = gij;
    });
}

template <typename scalar_t>
DDStiffnessMatrix<scalar_t>::DDStiffnessMatrix(const H1Space2D &fem, const EnsembleSpace &efem)
    : n_basis(fem.basis().size()),
      mx_elem(efem.max_n_elem()),
      n_domains(efem.size()),
      d(n_basis * n_basis),
      g(n_basis * n_basis * mx_elem * n_domains)
{
    make_diffmat<scalar_t>(d.host_write(), fem.basis());
    geom_factors<scalar_t>(g.device_write(), fem, efem);
    d_I = efem.subspace_indices(MemorySpace::DEVICE);
}

namespace cuddh
{
    template class DDStiffnessMatrix<float>;
    template class DDStiffnessMatrix<double>;
} // namespace cuddh
