#include "DD2D/DDStiffnessMatrix.hpp"

#include "SparseMatrix.hpp"

using namespace cuddh;

template <typename scalar_t>
static void make_diffmat(scalar_t *h_D, const Basis &basis)
{
    auto D = basis.derivative_matrix();
    for (int i = 0; i < D.size(); ++i)
        h_D[i] = scalar_t(D[i]);
}

template <typename scalar_t>
static void geom_factors(SmallSymmetricMatrix<scalar_t, 2> *d_G, const EnsembleSpace &efem)
{
    const Mesh2D &mesh = efem.h1_space().mesh();
    const Basis &basis = efem.h1_space().basis();
    const QuadratureRule &q = basis.quadrature();

    const int n_basis = basis.size();
    const int n_domains = efem.size();
    const int mx_elem = efem.max_n_elem();

    auto w = q.w(MemorySpace::DEVICE);
    auto x = q.x(MemorySpace::DEVICE);

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

        const double2x2 J = element.jacobian({x(i), x(j)});
        const double W = w(i) * w(j) / det(J);

        SmallSymmetricMatrix<scalar_t, 2> gij;
        gij(0, 0) = W * (J(1, 1) * J(1, 1) + J(0, 1) * J(0, 1));
        gij(1, 0) = -W * (J(1, 0) * J(1, 1) + J(0, 0) * J(0, 1));
        gij(1, 1) = W * (J(1, 0) * J(1, 0) + J(0, 0) * J(0, 0));

        G(i, j, el, subsp) = gij;
    });
}

template <typename scalar_t>
DDStiffnessMatrix<scalar_t>::DDStiffnessMatrix(const EnsembleSpace &efem)
    : efem{efem},
      n_basis(efem.h1_space().basis().size()),
      mx_elem(efem.max_n_elem()),
      n_domains(efem.size()),
      d(n_basis * n_basis),
      g(n_basis * n_basis * mx_elem * n_domains)
{
    make_diffmat<scalar_t>(d.host_write(), efem.h1_space().basis());
    geom_factors<scalar_t>(g.device_write(), efem);
    d_I = efem.subspace_indices(MemorySpace::DEVICE);
}

// Element stiffness formula derived from the action: for output node (tx,ty) and input node (a,b),
//
//   K[(tx,ty),(a,b)] = delta(b,ty) * Σᵢ D(i,tx)·G(i,ty).xx·D(i,a)    [xx term]
//                    + D(a,tx)·G(a,ty).xy·D(ty,b)                      [first xy term]
//                    + D(b,ty)·G(tx,b).xy·D(tx,a)                      [second xy term]
//                    + delta(a,tx) * Σᵢ D(i,ty)·G(tx,i).yy·D(i,b)    [yy term]
//
// D(i,j) = h_D[i + n_basis*j]  (column-major TensorWrapper layout)
// G(i,j,el,p) = h_G[i + n_basis*(j + n_basis*(el + mx_elem*p))]
template <typename scalar_t>
void DDStiffnessMatrix<scalar_t>::assemble(scalar_t c, BlockSparseMatrix<scalar_t, false> &B) const
{
    const auto sI = efem.subspace_indices(MemorySpace::HOST);
    const auto nel = efem.n_elems(MemorySpace::HOST);

    auto D = reshape(d.host_read(), n_basis, n_basis);
    auto G = reshape(g.host_read(), n_basis, n_basis, mx_elem, n_domains);

    for (int p = 0; p < n_domains; ++p)
        for (int el = 0; el < nel(p); ++el)
            for (int tx = 0; tx < n_basis; ++tx)
                for (int ty = 0; ty < n_basis; ++ty)
                {
                    const int row = sI(tx, ty, el, p);
                    for (int a = 0; a < n_basis; ++a)
                        for (int b = 0; b < n_basis; ++b)
                        {
                            const int col = sI(a, b, el, p);
                            scalar_t K = 0;

                            if (b == ty)
                                for (int i = 0; i < n_basis; ++i)
                                    K += D(i, tx) * G(i, ty, el, p)(0, 0) * D(i, a);

                            K += D(a, tx) * G(a, ty, el, p)(0, 1) * D(ty, b);
                            K += D(b, ty) * G(tx, b, el, p)(0, 1) * D(tx, a);

                            if (a == tx)
                                for (int i = 0; i < n_basis; ++i)
                                    K += D(i, ty) * G(tx, i, el, p)(1, 1) * D(i, b);

                            B.set_value(p, row, col, c * K);
                        }
                }
}

template <typename scalar_t>
void DDStiffnessMatrix<scalar_t>::assemble(std::complex<scalar_t> c, BlockSparseMatrix<scalar_t, true> &B) const
{
    const auto sI = efem.subspace_indices(MemorySpace::HOST);
    const auto nel = efem.n_elems(MemorySpace::HOST);

    auto D = reshape(d.host_read(), n_basis, n_basis);
    auto G = reshape(g.host_read(), n_basis, n_basis, mx_elem, n_domains);

    for (int p = 0; p < n_domains; ++p)
        for (int el = 0; el < nel(p); ++el)
            for (int tx = 0; tx < n_basis; ++tx)
                for (int ty = 0; ty < n_basis; ++ty)
                {
                    const int row = sI(tx, ty, el, p);
                    for (int a = 0; a < n_basis; ++a)
                        for (int b = 0; b < n_basis; ++b)
                        {
                            const int col = sI(a, b, el, p);
                            scalar_t K = 0;

                            if (b == ty)
                                for (int i = 0; i < n_basis; ++i)
                                    K += D(i, tx) * G(i, ty, el, p)(0, 0) * D(i, a);

                            K += D(a, tx) * G(a, ty, el, p)(0, 1) * D(ty, b);
                            K += D(b, ty) * G(tx, b, el, p)(0, 1) * D(tx, a);

                            if (a == tx)
                                for (int i = 0; i < n_basis; ++i)
                                    K += D(i, ty) * G(tx, i, el, p)(1, 1) * D(i, b);

                            B.set_value(p, row, col, c * K);
                        }
                }
}

namespace cuddh
{
    template class DDStiffnessMatrix<float>;
    template class DDStiffnessMatrix<double>;
} // namespace cuddh
