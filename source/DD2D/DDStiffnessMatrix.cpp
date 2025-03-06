#include "DD2D/DDStiffnessMatrix.hpp"

using namespace cuddh;

static void make_diffmat(float *h_D, const Basis &basis)
{
    const int n_basis = basis.size();
    dmat D(n_basis, n_basis);
    basis.deriv(n_basis, basis.quadrature().x(), D);
    for (int i = 0; i < n_basis * n_basis; ++i)
        h_D[i] = D[i];
}

static void geom_factors(float3 *d_G, const H1Space2D &fem, const EnsembleSpace &efem)
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

    const double *d_J = mesh.element_metrics(q).jacobians(MemorySpace::DEVICE);
    auto J = reshape(d_J, 2, 2, n_basis, n_basis, mesh.n_elem());

    auto n_elems = efem.n_elems(MemorySpace::DEVICE);
    auto elems = efem.elements(MemorySpace::DEVICE);

    auto G = reshape(d_G, n_basis, n_basis, mx_elem, n_domains);

    forall_3d(n_basis, n_basis, mx_elem, n_domains, [=] __device__ (int subsp) mutable -> void
    {
        const int n_elem = n_elems[subsp];

        const int i = threadIdx.x;
        const int j = threadIdx.y;
        const int el = threadIdx.z;

        if (el >= n_elem)
            return;

        const int g_el = elems(el, subsp);

        const double W = w(i) * w(j);
        const double Y_eta = J(1, 1, i, j, g_el);
        const double X_eta = J(0, 1, i, j, g_el);
        const double Y_xi  = J(1, 0, i, j, g_el);
        const double X_xi  = J(0, 0, i, j, g_el);

        const double detJ = X_xi * Y_eta - X_eta * Y_xi;
        
        float3 gij;
        gij.x =  W * (Y_eta * Y_eta + X_eta * X_eta) / detJ;
        gij.y = -W * (Y_xi  * Y_eta + X_xi  * X_eta) / detJ;
        gij.z =  W * (Y_xi  * Y_xi  + X_xi  * X_xi)  / detJ;

        G(i, j, el, subsp) = gij;
    });
}

DDStiffnessMatrix::DDStiffnessMatrix(const H1Space2D &fem, const EnsembleSpace &efem)
    : n_basis(fem.basis().size()),
      mx_elem(efem.max_n_elem()),
      n_domains(efem.size()),
      d(n_basis * n_basis),
      g(n_basis * n_basis * mx_elem * n_domains)
{
    make_diffmat(d.host_write(), fem.basis());
    geom_factors(g.device_write(), fem, efem);
}