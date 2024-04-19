#include "StiffnessMatrix.hpp"

namespace cuddh
{
    StiffnessMatrix::StiffnessMatrix(const H1Space& fem)
        : ndof{fem.size()},
          n_elem{fem.mesh().n_elem()},
          n_basis{fem.basis().size()},
          n_quad{fem.mesh().max_element_order() + n_basis},
          quad(n_quad, QuadratureRule::GaussLegendre),
          P(n_quad, n_basis),
          D(n_quad, n_basis),
          J(fem.mesh().element_metrics(quad).jacobians(), 2, 2, n_quad, n_quad, n_elem),
          I{fem.global_indices()}
    {
        fem.basis().eval(n_quad, quad.x(), P);
        fem.basis().deriv(n_quad, quad.x(), D);
    }

    StiffnessMatrix::StiffnessMatrix(const H1Space& fem, const QuadratureRule& quad_)
        : ndof{fem.size()},
          n_elem{fem.mesh().n_elem()},
          n_basis{fem.basis().size()},
          n_quad{quad_.size()},
          quad{quad_},
          P(n_quad, n_basis),
          D(n_quad, n_basis),
          J(fem.mesh().element_metrics(quad).jacobians(), 2, 2, n_quad, n_quad, n_elem),
          I{fem.global_indices()}
    {
        fem.basis().eval(n_quad, quad.x(), P);
        fem.basis().deriv(n_quad, quad.x(), D);
    }

    void StiffnessMatrix::action(double c, const double * x, double * y) const
    {
        dmat u(n_quad, n_quad);
        dmat Pu(n_quad, n_basis);
        dmat Du(n_quad, n_basis);
        dcube F(2, n_quad, n_quad);

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
            for (int l = 0; l < n_basis; ++l)
            {
                for (int i = 0; i < n_quad; ++i)
                {
                    double pxu = 0.0, dxu = 0.0;
                    for (int k = 0; k < n_basis; ++k)
                    {
                        const double uk = u(k, l);
                        pxu += P(i, k) * uk;
                        dxu += D(i, k) * uk;
                    }
                    Pu(i, l) = pxu;
                    Du(i, l) = dxu;
                }
            }

            for (int j = 0; j < n_quad; ++j)
            {
                for (int i = 0; i < n_quad; ++i)
                {
                    const double W = quad.w(i) * quad.w(j);
                    const double Y_eta = J(1, 1, i, j, el);
                    const double X_eta = J(0, 1, i, j, el);
                    const double Y_xi  = J(1, 0, i, j, el);
                    const double X_xi  = J(0, 0, i, j, el);

                    const double detJ = X_xi * Y_eta - X_eta * Y_xi;
                    const double A =  W * (Y_eta * Y_eta + X_eta * X_eta) / detJ;
                    const double B = -W * (Y_xi  * Y_eta + X_xi  * X_eta) / detJ;
                    const double C =  W * (Y_xi  * Y_xi  + X_xi  * X_xi)  / detJ;

                    double Dx = 0.0, Dy = 0.0;
                    for (int l = 0; l < n_basis; ++l)
                    {
                        Dx += P(j, l) * Du(i, l);
                        Dy += D(j, l) * Pu(i, l);
                    }
                    F(0, i, j) = A * Dx + B * Dy;
                    F(1, i, j) = B * Dx + C * Dy;
                }
            }

            // integrate
            for (int k = 0; k < n_basis; ++k)
            {
                for (int j = 0; j < n_quad; ++j)
                {
                    double df = 0.0, pg = 0.0;
                    for (int i = 0; i < n_quad; ++i)
                    {
                        df += D(i, k) * F(0, i, j);
                        pg += P(i, k) * F(1, i, j);
                    }
                    Du(j, k) = df;
                    Pu(j, k) = pg;
                }
            }

            for (int l = 0; l < n_basis; ++l)
            {
                for (int k = 0; k < n_basis; ++k)
                {
                    double Su = 0.0;
                    for (int j = 0; j < n_quad; ++j)
                    {
                        Su += P(j, l) * Du(j, k) + D(j, l) * Pu(j, k);
                    }
                    y[I(k, l, el)] += c * Su;
                }
            }
        }
    }

    void StiffnessMatrix::action(const double * x, double * y) const
    {
        for (int i = 0; i < ndof; ++i)
            y[i] = 0.0;
        action(1.0, x, y);
    }
} // namespace cuddh
