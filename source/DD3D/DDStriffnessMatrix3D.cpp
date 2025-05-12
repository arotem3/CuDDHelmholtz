#include "DD3D/DDStiffnessMatrix3D.hpp"

using namespace cuddh;

static void make_diffmat(float *h_D, const Basis &basis)
{
    const int n_basis = basis.size();
    dmat D(n_basis, n_basis);
    basis.deriv(n_basis, basis.quadrature().x(), D);
    for (int i = 0; i < n_basis * n_basis; ++i)
        h_D[i] = D[i];
}

static void geom_factors(fsym3x3 *d_G, const H1Space3D &fem, const EnsembleSpace3D &efem)
{
    const DeviceMesh3D &mesh = fem.mesh().to_device();
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

    host_device_dvec _x(n_basis);
    double *h_x = _x.host_write();
    for (int i = 0; i < n_basis; ++i)
        h_x[i] = q.x(i);
    auto x = reshape(_x.device_read(), n_basis);

    auto n_elems = efem.n_elems(MemorySpace::DEVICE);
    auto elems = efem.elements(MemorySpace::DEVICE);

    auto G = reshape(d_G, n_basis, n_basis, n_basis, mx_elem, n_domains);

    forall_3d(n_basis, n_basis, mx_elem, n_domains, [=] __device__ (int subsp) mutable -> void
    {
        const int n_elem = n_elems[subsp];

        const int i = threadIdx.x;
        const int j = threadIdx.y;
        const int el = threadIdx.z;

        if (el >= n_elem)
            return;

        const int g_el = elems(el, subsp);
        const HexElement element = mesh.element(g_el);

        double3 xi{x(i), x(j), 0.0};

        for (int k = 0; k < n_basis; ++k)
        {
            xi.z = x(k);
            
            double3x3 J = element.jacobian(xi);
            const double s = w(i) * w(j) * w(k) / det(J);

            J = adjugate(J);

            fsym3x3 g;

            #pragma unroll
            for (int m = 0; m < 3; ++m)
            {
                #pragma unroll
                for (int n = 0; n < m; ++n)
                {
                    double gmn = 0.0;
                    for (int l = 0; l < 3; ++l)
                        gmn += J(m, l) * J(n, l);
                    g(m, n) = s * gmn;
                }

                double gmm = 0.0;
                for (int l = 0; l < 3; ++l)
                    gmm += J(m, l) * J(m, l);
                g(m, m) = s * gmm;
            }

            G(i, j, k, el, subsp) = g;
        }
    });
}

DDStiffnessMatrix3D::DDStiffnessMatrix3D(const H1Space3D &fem, const EnsembleSpace3D &efem)
    : n_basis(fem.basis().size()),
      mx_elem(efem.max_n_elem()),
      n_domains(efem.size()),
      d(n_basis * n_basis),
      g(n_basis * n_basis * mx_elem * n_domains)
{
    make_diffmat(d.host_write(), fem.basis());
    geom_factors(g.device_write(), fem, efem);
}