#include "DD2D/DDFaceMassMatrix.hpp"

using namespace cuddh;

template <typename scalar_t>
DDFaceMassMatrix<scalar_t>::DDFaceMassMatrix(const H1Space2D &fem, const EnsembleSpace &efem)
    : mx_fdof(efem.max_fsize()), n_domains(efem.size()), m(mx_fdof * n_domains)
{
    const Mesh2D &mesh = fem.mesh();
    const Basis &basis = fem.basis();
    const QuadratureRule &q = basis.quadrature();

    const int n_basis = basis.size();

    auto n_faces = efem.n_faces(MemorySpace::HOST);
    auto faces = efem.faces(MemorySpace::HOST);
    auto f_inds = efem.face_indices(MemorySpace::HOST);

    auto H = reshape(m.host_write(), mx_fdof, n_domains);

    for (int subsp = 0; subsp < n_domains; ++subsp)
    {
        const int s_nf = n_faces(subsp);
        for (int f = 0; f < s_nf; ++f)
        {
            const int g_f = faces(f, subsp);
            const Edge *edge = mesh.edge(g_f);

            for (int i = 0; i < n_basis; ++i)
            {
                const double xi = q.x(i);
                const double ds = edge->measure(xi);

                const int l = f_inds(i, f, subsp);
                H(l, subsp) += static_cast<scalar_t>(ds * q.w(i));
            }
        }
    }
}

namespace cuddh
{
    template class DDFaceMassMatrix<float>;
    template class DDFaceMassMatrix<double>;
} // namespace cuddh
