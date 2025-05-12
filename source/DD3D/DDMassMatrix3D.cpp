#include "DD3D/DDMassMatrix3D.hpp"

using namespace cuddh;

static void mass(float *d_m, const H1Space3D &fem, const EnsembleSpace3D &efem)
{
    const DeviceMesh3D &mesh = fem.mesh().to_device();
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

    host_device_dvec _x(n_basis);
    double *h_x = _x.host_write();
    for (int i = 0; i < n_basis; ++i)
        h_x[i] = q.x(i);
    auto x = reshape(_x.device_read(), n_basis);

    auto d_n_elems = efem.n_elems(MemorySpace::DEVICE);
    auto d_elems = efem.elements(MemorySpace::DEVICE);
    auto sI = efem.subspace_indices(MemorySpace::DEVICE);

    auto M = reshape(d_m, mx_dofs, n_domains);

    forall_3d(n_basis, n_basis, mx_elem_per_dom, n_domains, [=] __device__ (int subsp) mutable
    {
        const int s_nel = d_n_elems(subsp);
        
        const int i = threadIdx.x;
        const int j = threadIdx.y;
        const int el = threadIdx.z;

        if (el < s_nel)
        {
            const int g_el = d_elems(el, subsp);
            const HexElement element = mesh.element(g_el);
            
            double3 xi{x(i), x(j), 0.0};

            for (int k = 0; k < n_basis; ++k)
            {
                const int l = sI(i, j, k, el, subsp);

                xi.z = x(k);
                float val = w(i) * w(j) * w(k) * element.measure(xi);
                atomicAdd(&M(l, subsp), val);
            }
        }
    });
}

DDMassMatrix3D::DDMassMatrix3D(const H1Space3D &fem, const EnsembleSpace3D &efem)
    : mx_dofs(efem.max_size()),
      n_domains(efem.size()),
      m(mx_dofs * n_domains)
{
    mass(m.device_write(), fem, efem);
}
