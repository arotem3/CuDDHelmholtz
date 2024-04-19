#include "FaceMassMatrix.hpp"

namespace cuddh
{
    FaceMassMatrix::FaceMassMatrix(const FaceSpace& fs)
        : ndof{fs.size()},
          n_faces{fs.n_faces()},
          n_basis{fs.h1_space().basis().size()},
          n_quad{fs.h1_space().mesh().max_element_order() + n_basis},
          quad(n_quad, QuadratureRule::GaussLegendre),
          I{fs.subspace_indices()},
          P(n_quad, n_basis)
    {
        auto& metrics = fs.metrics(quad);
        detJ = reshape(metrics.measures(), n_quad, n_faces);

        fs.h1_space().basis().eval(n_quad, quad.x(), P);
    }

    void FaceMassMatrix::action(double c, const double * x, double * y) const
    {
        dvec u(n_basis);
        dvec Pu(n_quad);

        for (int f = 0; f < n_faces; ++f)
        {
            // copy faces values to u
            for (int k = 0; k < n_basis; ++k)
            {
                const int idx = I(k, f);
                u(k) = x[idx];
            }

            // eval on quadrature rule and scale by jacobian
            for (int i = 0; i < n_quad; ++i)
            {
                double pu = 0.0;
                for (int k = 0; k < n_basis; ++k)
                {
                    pu += P(i, k) * u(k);
                }
                Pu(i) = pu * quad.w(i) * detJ(i, f);
            }

            // integrate
            for (int k = 0; k < n_basis; ++k)
            {
                double Mu = 0.0;
                for (int i = 0; i < n_quad; ++i)
                {
                    Mu += P(i, k) * Pu(i);
                }

                y[I(k, f)] += c * Mu;
            }
        }
    }

    void FaceMassMatrix::action(const double * x, double * y) const
    {
        for (int i = 0; i < ndof; ++i)
            y[i] = 0.0;
        action(1.0, x, y);
    }

    DiagInvFaceMassMatrix::DiagInvFaceMassMatrix(const FaceSpace& fs)
        : ndof(fs.size()),
          inv_m(ndof)
    {
        const int nf = fs.n_faces();
        auto& q = fs.h1_space().basis().quadrature();
        const int n_basis = q.size();

        auto& metrics = fs.metrics(q);
        auto detJ = reshape(metrics.measures(), n_basis, nf);

        auto I = fs.subspace_indices();

        for (int f = 0; f < nf; ++f)
        {
            for (int i = 0; i < n_basis; ++i)
            {
                inv_m(I(i, f)) += q.w(i) * detJ(i, f);
            }
        }

        for (int i = 0; i < ndof; ++i)
            inv_m(i) = 1.0 / inv_m(i);
    }

    void DiagInvFaceMassMatrix::action(double c, const double * x, double * y) const
    {
        for (int i = 0; i < ndof; ++i)
            y[i] += c * inv_m(i) * x[i];
    }

    void DiagInvFaceMassMatrix::action(const double * x, double * y) const
    {
        for (int i = 0; i < ndof; ++i)
            y[i] = inv_m(i) * x[i];
    }
} // namespace cuddh
