#include "DD3D/DDStiffnessMatrix3D.hpp"

#include "forall.hpp"

using namespace cuddh;

template <typename scalar_t>
static thrust::device_vector<scalar_t> make_diffmat(const Basis &basis)
{
    const int n_basis = basis.size();

    thrust::host_vector<double> D(n_basis * n_basis);
    double *Dptr = thrust::raw_pointer_cast(D.data());
    basis.deriv(n_basis, basis.quadrature().x(MemorySpace::HOST), Dptr);

    return D;
}

template <typename scalar_t>
static thrust::device_vector<SmallSymmetricMatrix<scalar_t, 3>> geom_factors(const EnsembleSpace3D &efem)
{
    using mat_t = SmallSymmetricMatrix<scalar_t, 3>;

    const DeviceMesh3D &mesh = efem.h1_space().mesh().to_device();
    const Basis &basis = efem.h1_space().basis();
    const QuadratureRule &q = basis.quadrature();

    const int n_basis = basis.size();
    const int n_domains = efem.size();
    const int mx_elem = efem.max_n_elem();

    auto w = q.w(MemorySpace::DEVICE);
    auto x = q.x(MemorySpace::DEVICE);

    auto n_elems = efem.n_elems(MemorySpace::DEVICE);
    auto elems = efem.elements(MemorySpace::DEVICE);

    thrust::device_vector<mat_t> d_G(n_basis * n_basis * n_basis * mx_elem * n_domains);
    auto G = reshape(thrust::raw_pointer_cast(d_G.data()), n_basis, n_basis, n_basis, mx_elem, n_domains);

    forall_2d(n_basis, n_basis, mx_elem * n_domains, [=] __device__(int b) mutable -> void {
        const int el = b % mx_elem;
        const int subsp = b / mx_elem;
        const int i = threadIdx.x, j = threadIdx.y;

        if (el >= n_elems(subsp))
            return;

        const int g_el = elems(el, subsp);

        __shared__ HexElement element;
        if (i == 0 && j == 0)
            element = mesh.element(g_el);
        __syncthreads();

        double3 xi{x(i), x(j), 0.0};

        for (int k = 0; k < n_basis; ++k)
        {
            xi.z = x(k);

            double3x3 J = element.jacobian(xi);
            const double s = w(i) * w(j) * w(k) / det(J);

            J = adjugate(J);

            mat_t g;

            for (int m = 0; m < 3; ++m)
            {
                for (int n = 0; n <= m; ++n)
                {
                    double gmn = 0.0;
                    for (int l = 0; l < 3; ++l)
                        gmn += J(m, l) * J(n, l);
                    g(m, n) = s * gmn;
                }
            }

            G(i, j, k, el, subsp) = g;
        }
    });

    return d_G;
}

template <typename scalar_t>
DDStiffnessMatrix3D<scalar_t>::DDStiffnessMatrix3D(const EnsembleSpace3D &efem)
    : n_basis(efem.h1_space().basis().size()), mx_elem(efem.max_n_elem()), n_domains(efem.size())
{
    d = make_diffmat<scalar_t>(efem.h1_space().basis());
    g = geom_factors<scalar_t>(efem);
    d_I = efem.subspace_indices(MemorySpace::DEVICE);
}

template class DDStiffnessMatrix3D<float>;
template class DDStiffnessMatrix3D<double>;
