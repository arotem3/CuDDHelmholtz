#include "DD3D/DDMassMatrix3D.hpp"

#include "forall.hpp"

using namespace cuddh;

template <typename scalar_t>
static thrust::universal_vector<scalar_t> mass(const EnsembleSpace3D &efem, const GridFunc3D<double> *a)
{
    const H1Space3D &fem = efem.h1_space();
    const DeviceMesh3D &mesh = fem.mesh().to_device();
    const Basis &basis = fem.basis();
    const QuadratureRule &q = basis.quadrature();

    const int n_basis = basis.size();
    const int mx_elem_per_dom = efem.max_n_elem();
    const int n_domains = efem.size();
    const int mx_dofs = efem.max_size();

    auto w = q.w(MemorySpace::DEVICE);
    auto x = q.x(MemorySpace::DEVICE);

    auto d_n_elems = efem.n_elems(MemorySpace::DEVICE);
    auto d_elems = efem.elements(MemorySpace::DEVICE);
    auto sI = efem.subspace_indices(MemorySpace::DEVICE);

    TensorWrapper<4, const double> A;
    if (a)
        A = a->read(MemorySpace::DEVICE);

    thrust::universal_vector<scalar_t> u_m(mx_dofs * n_domains, scalar_t(0));
    auto M = reshape(u_m, mx_dofs, n_domains);

    forall_2d(n_basis, n_basis, mx_elem_per_dom * n_domains, [=] __device__(int b) mutable {
        const int el = b % mx_elem_per_dom;
        const int subsp = b / mx_elem_per_dom;

        if (el >= d_n_elems(subsp))
            return;

        const int i = threadIdx.x;
        const int j = threadIdx.y;

        const int g_el = d_elems(el, subsp);

        __shared__ HexElement element;
        if (i == 0 && j == 0)
            element = mesh.element(g_el);
        __syncthreads();

        double3 xi{x(i), x(j), 0.0};

        for (int k = 0; k < n_basis; ++k)
        {
            const int l = sI(i, j, k, el, subsp);

            xi.z = x(k);
            scalar_t val = static_cast<scalar_t>(w(i) * w(j) * w(k) * element.measure(xi));

            if (A)
                val *= static_cast<scalar_t>(A(i, j, k, g_el));

            atomicAdd(&M(l, subsp), val);
        }
    });

    return u_m;
}

template <typename scalar_t>
DDMassMatrix3D<scalar_t>::DDMassMatrix3D(const EnsembleSpace3D &efem) : mx_dofs(efem.max_size()), n_domains(efem.size())
{
    m = mass<scalar_t>(efem, nullptr);
}

template <typename scalar_t>
DDMassMatrix3D<scalar_t>::DDMassMatrix3D(const EnsembleSpace3D &efem, const GridFunc3D<double> &a)
    : mx_dofs(efem.max_size()), n_domains(efem.size())
{
    m = mass<scalar_t>(efem, &a);
}

namespace cuddh
{
    template class DDMassMatrix3D<float>;
    template class DDMassMatrix3D<double>;
} // namespace cuddh
