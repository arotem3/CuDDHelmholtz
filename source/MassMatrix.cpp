#include "MassMatrix.hpp"

namespace cuddh
{
    static void init_mass_matrix(int n_elem,
                                 int n_basis,
                                 int n_quad,
                                 const double * a,
                                 const double * detJ_,
                                 const QuadratureRule& quad,
                                 const_icube_wrapper I,
                                 const dmat& P,
                                 double * op_)
    {
        auto detJ = reshape(detJ_, n_quad, n_quad, n_elem);
        auto op = reshape(op_, n_quad, n_quad, n_elem);

        dmat x(n_quad, n_quad);
        dmat z(n_quad, n_basis);

        for (int el = 0; el < n_elem; ++el)
        {
            // copy global dofs to element
            for (int l = 0; l < n_basis; ++l)
            {
                for (int k = 0; k < n_basis; ++k)
                {
                    const int idx = I(k, l, el);
                    x(k, l) = (a) ? a[idx] : 1.0;
                }
            }

            // evaluate on quadrature points
            for (int i = 0; i < n_quad; ++i)
            {
                for (int l = 0; l < n_basis; ++l)
                {
                    double px = 0.0;
                    for (int k = 0; k < n_basis; ++k)
                    {
                        px += P(i, k) * x(k, l);
                    }
                    z(i, l) = px;
                }
            }

            for (int i = 0; i < n_quad; ++i)
            {
                for (int j = 0; j < n_quad; ++j)
                {
                    double ppx = 0.0;
                    for (int l = 0; l < n_basis; ++l)
                    {
                        ppx += P(j, l) * z(i, l);
                    }
                    op(i, j, el) = detJ(i, j, el) * quad.w(i) * quad.w(j) * ppx;
                }
            }
        }
    }

    MassMatrix::MassMatrix(const H1Space& fem)
        : n_elem{fem.mesh().n_elem()},
          n_basis{fem.basis().size()},
          n_quad{n_basis + fem.mesh().max_element_order()},
          quad(n_quad, QuadratureRule::GaussLegendre),
          P(n_quad, n_basis),
          a(n_quad, n_quad, n_elem),
          I{fem.global_indices()}
    {
        const double * detJ = fem.mesh().element_metrics(quad).measures();

        fem.basis().eval(n_quad, quad.x(), P);

        init_mass_matrix(n_elem, n_basis, n_quad, nullptr, detJ, quad, I, P, a);
    }

    MassMatrix::MassMatrix(const double * a_, const H1Space& fem)
        : n_elem{fem.mesh().n_elem()},
          n_basis{fem.basis().size()},
          n_quad(1 + 1.5*n_basis + fem.mesh().max_element_order()),
          quad(n_quad, QuadratureRule::GaussLegendre),
          P(n_quad, n_basis),
          a(n_quad, n_quad, n_elem),
          I{fem.global_indices()}
    {
        const double * detJ = fem.mesh().element_metrics(quad).measures();

        fem.basis().eval(n_quad, quad.x(), P);

        init_mass_matrix(n_elem, n_basis, n_quad, a_, detJ, quad, I, P, a);
    }

    void MassMatrix::action(double c, const double * x, double * y) const
    {
        dmat u(n_quad, n_quad);
        dmat Pu(n_quad, n_basis);

        for (int el = 0; el < n_elem; ++el)
        {
            // copy global dofs to element
            for (int l = 0; l < n_basis; ++l)
            {
                for (int k = 0; k < n_basis; ++k)
                {
                    u(k, l) = x[I(k, l, el)];
                }
            }

            // evaluate on quadrature points
            for (int i = 0; i < n_quad; ++i)
            {
                for (int l = 0; l < n_basis; ++l)
                {
                    double pu = 0.0;
                    for (int k = 0; k < n_basis; ++k)
                    {
                        pu += P(i, k) * u(k, l);
                    }
                    Pu(i, l) = pu;
                }
            }

            for (int i = 0; i < n_quad; ++i)
            {
                for (int j = 0; j < n_quad; ++j)
                {
                    double ppu = 0.0;
                    for (int l = 0; l < n_basis; ++l)
                    {
                        ppu += P(j, l) * Pu(i, l);
                    }
                    u(i, j) = a(i, j, el) * ppu;
                }
            }

            // integrate
            for (int i = 0; i < n_quad; ++i)
            {
                for (int l = 0; l < n_basis; ++l)
                {
                    double qu = 0.0;
                    for (int j = 0; j < n_quad; ++j)
                    {
                        qu += P(j, l) * u(i, j);
                    }
                    Pu(i, l) = qu;
                }
            }

            for (int k = 0; k < n_basis; ++k)
            {
                for (int l = 0; l < n_basis; ++l)
                {
                    double qqu = 0.0;
                    for (int i = 0; i < n_quad; ++i)
                    {
                        qqu += P(i, k) * Pu(i, l);
                    }
                    u(k, l) = qqu;
                }
            }

            // map back to global
            for (int l = 0; l < n_basis; ++l)
            {
                for (int k = 0; k < n_basis; ++k)
                {
                    y[I(k, l, el)] += c * u(k, l);
                }
            }
        }
    }

    DiagInvMassMatrix::DiagInvMassMatrix(const H1Space& fem)
        : ndof{fem.size()},
          p(ndof),
          I{fem.global_indices()}
    {
        const int n_elem = fem.mesh().n_elem();
        const int n_basis = fem.basis().size();

        auto& q = fem.basis().quadrature();
        auto& metrics = fem.mesh().element_metrics(q);
        auto detJ = reshape(metrics.measures(), n_basis, n_basis, n_elem);

        for (int el = 0; el < n_elem; ++el)
        {
            for (int j = 0; j < n_basis; ++j)
            {
                for (int i = 0; i < n_basis; ++i)
                {
                    p(I(i, j, el)) += q.w(i) * q.w(j) * detJ(i, j, el);
                }
            }
        }

        for (int i = 0; i < ndof; ++i)
            p(i) = 1.0 / p(i);
    }

    DiagInvMassMatrix::DiagInvMassMatrix(const double * a, const H1Space& fem)
        : ndof(fem.size()),
          p(ndof),
          I{fem.global_indices()}
    {
        const int n_elem = fem.mesh().n_elem();
        const int n_basis = fem.basis().size();

        auto& q = fem.basis().quadrature();
        auto& metrics = fem.mesh().element_metrics(q);
        auto detJ = reshape(metrics.measures(), n_basis, n_basis, n_elem);

        for (int el = 0; el < n_elem; ++el)
        {
            for (int j = 0; j < n_basis; ++j)
            {
                for (int i = 0; i < n_basis; ++i)
                {
                    const int idx = I(i, j, el);
                    p(idx) += q.w(i) * q.w(j) * detJ(i, j, el) * a[idx];
                }
            }
        }

        for (int i = 0; i < ndof; ++i)
            p(i) = 1.0 / p(i);
    }

    void DiagInvMassMatrix::action(double c, const double * x, double * y) const
    {
        for (int i = 0; i < ndof; ++i)
            y[i] += c * p(i) * x[i];
    }
} // namespace cuddh
