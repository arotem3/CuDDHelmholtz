#include "DD2D/DDMassMatrix.hpp"

using namespace cuddh;

template <typename scalar_t>
static HostDeviceArray<scalar_t> init_mass(const EnsembleSpace &efem, const GridFunc2D<double> *_a)
{
    const Mesh2D &mesh = efem.h1_space().mesh();
    const Basis &basis = efem.h1_space().basis();
    const QuadratureRule &q = basis.quadrature();

    const int n_basis = basis.size();
    const int mx_elem_per_dom = efem.max_n_elem();
    const int n_domains = efem.size();
    const int mx_dofs = efem.max_size();

    auto w = q.w(MemorySpace::DEVICE);
    auto x = q.x(MemorySpace::DEVICE);

    auto d_mesh = mesh.to_device();

    auto d_n_elems = efem.n_elems(MemorySpace::DEVICE);
    auto d_elems = efem.elements(MemorySpace::DEVICE);
    auto sI = efem.subspace_indices(MemorySpace::DEVICE);

    TensorWrapper<3, const double> a;
    if (_a)
        a = _a->read(MemorySpace::DEVICE);

    HostDeviceArray<scalar_t> m(mx_dofs * n_domains);
    auto M = reshape(m.device_write(), mx_dofs, n_domains);

    forall_2d(n_basis, n_basis, mx_elem_per_dom * n_domains, [=] __device__(int index) mutable {
        const auto [i, j, _] = threadIdx;
        const int el = index % mx_elem_per_dom;
        const int subsp = index / mx_elem_per_dom;

        if (el >= d_n_elems(subsp))
            return;

        const int g_el = d_elems(el, subsp);

        __shared__ QuadElement element;
        if (i == 0 && j == 0)
            element = d_mesh.element(g_el);
        __syncthreads();

        scalar_t val = w(i) * w(j) * element.measure({x(i), x(j)});

        if (a)
            val *= a(i, j, g_el);

        int l = sI(i, j, el, subsp);
        atomicAdd(&M(l, subsp), val);
    });

    return m;
}

template <typename scalar_t>
DDMassMatrix<scalar_t>::DDMassMatrix(const EnsembleSpace &efem) : mx_dofs{efem.max_size()}, n_domains{efem.size()}
{
    m = init_mass<scalar_t>(efem, nullptr);
}

template <typename scalar_t>
DDMassMatrix<scalar_t>::DDMassMatrix(const EnsembleSpace &efem, const GridFunc2D<double> &a)
    : mx_dofs{efem.max_size()}, n_domains{efem.size()}
{
    m = init_mass<scalar_t>(efem, &a);
}

namespace cuddh
{
    template class DDMassMatrix<float>;
    template class DDMassMatrix<double>;
} // namespace cuddh
