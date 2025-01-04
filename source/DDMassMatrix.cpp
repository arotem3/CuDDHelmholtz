#include "DDMassMatrix.hpp"

using namespace cuddh;

DDMassMatrix::DDMassMatrix(const H1Space &fem, const EnsembleSpace &efem)
    : mx_dofs(efem.max_size()),
      n_domains(efem.size()),
      m(mx_dofs * n_domains)
{
    const Mesh2D &mesh = fem.mesh();
    const Basis &basis = fem.basis();
    const QuadratureRule &q = basis.quadrature();

    const int n_basis = basis.size();
    const int mx_elem_per_dom = efem.max_n_elem();

    auto h_n_elems = efem.n_elems(MemorySpace::HOST);
    auto h_elems = efem.elements(MemorySpace::HOST);
    auto sI = efem.subspace_indices(MemorySpace::HOST);

    const double *h_detJ = mesh.element_metrics(q).measures(MemorySpace::HOST);
    auto detJ = reshape(h_detJ, n_basis, n_basis, mesh.n_elem());

    auto M = reshape(m.host_write(), mx_dofs, n_domains);

    for (int subsp = 0; subsp < n_domains; ++subsp)
    {
        const int s_nel = h_n_elems(subsp);
        for (int el = 0; el < s_nel; ++el)
        {
            const int g_el = h_elems(el, subsp);
            for (int j = 0; j < n_basis; ++j)
            {
                for (int i = 0; i < n_basis; ++i)
                {
                    int l = sI(i, j, el, subsp);
                    M(l, subsp) += q.w(i) * q.w(j) * detJ(i, j, g_el);
                }
            }
        }
    }
}