#include "DD3D/DDStiffnessMatrix3D.hpp"

using namespace cuddh;

static thrust::universal_vector<float> make_diffmat(const Basis &basis)
{
    const int n_basis = basis.size();

    thrust::universal_vector<double> u_D(n_basis * n_basis);
    auto D = reshape(u_D, n_basis, n_basis);

    basis.deriv(n_basis, basis.quadrature().x(), D);

    return u_D; // auto-conversion to float
}

static thrust::universal_vector<fsym3x3> geom_factors(const H1Space3D &fem, const EnsembleSpace3D &efem)
{
    const DeviceMesh3D &mesh = fem.mesh().to_device();
    const Basis &basis = fem.basis();
    const QuadratureRule &q = basis.quadrature();

    const int n_basis = basis.size();
    const int n_domains = efem.size();
    const int mx_elem = efem.max_n_elem();

    thrust::universal_vector<double> u_w(n_basis);
    for (int i = 0; i < n_basis; ++i)
        u_w[i] = q.w(i);
    auto w = reshape(u_w, n_basis);

    thrust::universal_vector<double> u_x(n_basis);
    for (int i = 0; i < n_basis; ++i)
        u_x[i] = q.x(i);
    auto x = reshape(u_x, n_basis);

    auto n_elems = efem.n_elems(MemorySpace::DEVICE);
    auto elems = efem.elements(MemorySpace::DEVICE);

    thrust::universal_vector<fsym3x3> u_G(n_basis * n_basis * n_basis * mx_elem * n_domains);
    auto G = reshape(u_G, n_basis, n_basis, n_basis, mx_elem, n_domains);

    forall_3d(n_basis, n_basis, mx_elem, n_domains, [=] __device__(int subsp) mutable -> void {
        const auto [i, j, el] = threadIdx;

        if (el >= n_elems(subsp))
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

    return u_G;
}

DDStiffnessMatrix3D::DDStiffnessMatrix3D(const H1Space3D &fem, const EnsembleSpace3D &efem)
    : n_basis(fem.basis().size()), mx_elem(efem.max_n_elem()), n_domains(efem.size())
{
    d = make_diffmat(fem.basis());
    g = geom_factors(fem, efem);
}