#include "MassMatrix.hpp"

namespace cuddh
{
    template <int NQ>
    static void init_mass_matrix(int n_elem,
                                 int n_basis,
                                 int n_quad,
                                 const double * a, /* device */
                                 const double * detJ_, /* device */
                                 const QuadratureRule& quad,
                                 const int * I, /* device */
                                 const double * _P, /* device */
                                 double * op_ /* device */)
    {
        auto detJ = reshape(detJ_, n_quad, n_quad, n_elem);
        auto op = reshape(op_, n_quad, n_quad, n_elem);
        auto P_ = reshape(_P, n_quad, n_basis);
        auto I = reshape(I, n_basis, n_basis, n_elem);

        host_device_dvec _w(n_quad);
        double * h_w = _w.host_write();
        for (int i = 0; i < n_quad; ++i)
            h_w[i] = quad.w(i);
        
        auto w = reshape(_w.device_read(), n_quad);

        forall_2d(n_quad, n_quad, n_elem, [=] __device__ (int el) -> void {
            __shared__ double x[NQ][NQ];
            __shared__ double z[NQ][NQ];
            __shared__ double P[NQ][NQ];

            const int x = threadIdx.x;
            const int y = threadIdx.y;
            const int dx = blockDim.x;
            const int dy = blockDim.y;

            // copy P
            for (int j = y; j < n_quad; j += dy)
            {
                for (int i = x; i < n_quad; i+= dx)
                {
                    P[i][j] = P_(i, j);
                }
            }

            // copy global a to element
            for (int l = y; l < n_basis; l += dy)
            {
                for (int k = x; k < n_basis; l += dx)
                {
                    const idx = I(k, l, el);
                    x[l][k] = (a) ? a[idx] : 1.0;
                }
            }

            __syncthreads();

            // evaluate on quadrature points
            for (int i = x; i < n_quad; i += dx)
            {
                for (int l = y; l < n_basis; l += dy)
                {
                    double px = 0.0;
                    for (int k = 0; k < n_basis; ++k)
                    {
                        px += P[i][k] * x[l][k];
                    }
                    z[i][l] = px;
                }
            }

            __syncthreads();

            for (int j = x; j < n_quad; j += dx)
            {
                for (int i = y; i < n_quad; i += dy)
                {
                    double ppx = 0.0;
                    for (int l = 0; l < n_basis; ++l)
                    {
                        ppx += P[j][l] * z[i][l];
                    }
                    ppx *= w(i) * w(j) * detJ(i, j, el);

                    op(i, j, el) = ppx;
                }
            }
        });
    }

    MassMatrix::MassMatrix(const H1Space& fem)
        : ndof{fem.size()},
          n_elem{fem.mesh().n_elem()},
          n_basis{fem.basis().size()},
          n_quad{n_basis + fem.mesh().max_element_order()},
          quad(n_quad, QuadratureRule::GaussLegendre),
          _P(n_quad * n_basis),
          _a(n_quad * n_quad * n_elem),
          _I{fem.global_indices()}
    {
        auto const& detJ = fem.mesh().element_metrics(quad).measures();

        fem.basis().eval(n_quad, quad.x(), _P.host_write());

        if (n_quad <= 4)
            init_mass_matrix<4>(n_elem, n_basis, n_quad, nullptr, detJ.device_read(), quad, _I.device_read(), _P.device_read(), _a.device_write());
        else if (n_quad <= 8)
            init_mass_matrix<8>(n_elem, n_basis, n_quad, nullptr, detJ.device_read(), quad, _I.device_read(), _P.device_read(), _a.device_write());
        else if (n_quad <= 16)
            init_mass_matrix<16>(n_elem, n_basis, n_quad, nullptr, detJ.device_read(), quad, _I.device_read(), _P.device_read(), _a.device_write());
        else if (n_quad <= 24)
            init_mass_matrix<24>(n_elem, n_basis, n_quad, nullptr, detJ.device_read(), quad, _I.device_read(), _P.device_read(), _a.device_write());
        else if (n_quad <= 32)
            init_mass_matrix<32>(n_elem, n_basis, n_quad, nullptr, detJ.device_read(), quad, _I.device_read(), _P.device_read(), _a.device_write());
        else if (n_quad <= 64)
            init_mass_matrix<64>(n_elem, n_basis, n_quad, nullptr, detJ.device_read(), quad, _I.device_read(), _P.device_read(), _a.device_write());
        else
            cuddh_error("MassMatrix error: quadrature rules with more than 64 points not yet supported.");
    }

    MassMatrix::MassMatrix(const double * a_, const H1Space& fem)
        : ndof{fem.size()},
          n_elem{fem.mesh().n_elem()},
          n_basis{fem.basis().size()},
          n_quad(1 + 1.5*n_basis + fem.mesh().max_element_order()),
          quad(n_quad, QuadratureRule::GaussLegendre),
          _P(n_quad * n_basis),
          _a(n_quad * n_quad * n_elem),
          _I{fem.global_indices()}
    {
        auto const& detJ = fem.mesh().element_metrics(quad).measures();

        fem.basis().eval(n_quad, quad.x(), P.host_write());

        if (n_quad <= 4)
            init_mass_matrix<4>(n_elem, n_basis, n_quad, a_, detJ.device_read(), quad, _I.device_read(), _P.device_read(), _a.device_write());
        else if (n_quad <= 8)
            init_mass_matrix<8>(n_elem, n_basis, n_quad, a_, detJ.device_read(), quad, _I.device_read(), _P.device_read(), _a.device_write());
        else if (n_quad <= 16)
            init_mass_matrix<16>(n_elem, n_basis, n_quad, a_, detJ.device_read(), quad, _I.device_read(), _P.device_read(), _a.device_write());
        else if (n_quad <= 24)
            init_mass_matrix<24>(n_elem, n_basis, n_quad, a_, detJ.device_read(), quad, _I.device_read(), _P.device_read(), _a.device_write());
        else if (n_quad <= 32)
            init_mass_matrix<32>(n_elem, n_basis, n_quad, a_, detJ.device_read(), quad, _I.device_read(), _P.device_read(), _a.device_write());
        else if (n_quad <= 64)
            init_mass_matrix<64>(n_elem, n_basis, n_quad, a_, detJ.device_read(), quad, _I.device_read(), _P.device_read(), _a.device_write());
        else
            cuddh_error("MassMatrix error: quadrature rules with more than 64 points not yet supported.");
    }

    template <int NQ>
    static void mass_action(int n_elem,
                            int n_quad,
                            int n_basis,
                            const int * _I,
                            const double * _P,
                            const double * _a,
                            double c,
                            const double * x,
                            double * y)
    {
        auto I = reshape(_I, n_basis, n_basis, n_elem);
        auto P_ = reshape(_P, n_quad, n_basis);
        auto a = reshape(_a, n_quad, n_quad, n_elem);

        forall_2d(n_quad, n_quad, n_elem, [=] __device__ (int el) -> void {
            __shared__ double u[NQ][NQ];
            __shared__ double Pu[NQ][NQ];
            __shared__ double P[NQ][NQ];

            const int x = threadIdx.x;
            const int y = threadIdx.y;
            const int dx = blockDim.x;
            const int dy = blockDim.y;

            const int idx;

            // copy P
            for (int l = x; l < n_basis; l += dx)
            {
                for (int i = y; i < n_quad; i += dy)
                {
                    P[i][l] = P_(i, l);
                }
            }

            // copy global dofs to element
            for (int l = x; l < n_basis; l += dx)
            {
                for (int k = y; k < n_basis; k += dy)
                {
                    idx = I(k, l, el);
                    u[k][l] = x[idx];
                }
            }

            __syncthreads();

            // evaluate on quadrature points
            for (int i = x; i < n_quad; i += dx)
            {
                for (int l = y; l < n_basis; y += dy)
                {
                    double pu = 0.0;
                    for (int k = 0; k < n_basis; ++k)
                    {
                        pu += P[i][k] * u[k][l];
                    }
                    Pu[i][l] = pu;
                }
            }

            __syncthreads();

            for (int i = x; i < n_quad; i += dx)
            {
                for (int j = y; j < n_quad; j += dy)
                {
                    double ppu = 0.0;
                    for (int l = 0; l < n_basis; ++l)
                    {
                        ppu += P[j][l] * Pu[i][l];
                    }
                    u[i][j] = a(i, j, el) * ppu;
                }
            }

            __syncthreads();

            // integrate
            for (int i = x; i < n_quad; i += dx)
            {
                for (int l = y; l < n_basis; y += dy)
                {
                    double qu = 0.0;
                    for (int j = 0; j < n_quad; ++j)
                    {
                        qu += P[j][l] * u[i][j];
                    }
                    Pu[i][l] = qu;
                }
            }

            __syncthreads();

            for (int l = x; l < n_basis; l += dx)
            {
                for (int k = y; k < n_basis; k += dy)
                {
                    double qqu = 0.0;
                    for (int i = 0; i < n_quad; ++i)
                    {
                        qqu += P[i][k] * Pu[i][l];
                    }
                    qqu *= c;

                    AtomicAdd(y + idx, qqu);
                }
            }
        });
    }

    void MassMatrix::action(double c, const double * x, double * y) const
    {
        if (n_quad <= 4)
            mass_action<4>(n_elem, n_quad, n_basis, _I.device_read(), _P.device_read(), _a.device_read(), c, x, y);
        else if (n_quad <= 8)
            mass_action<8>(n_elem, n_quad, n_basis, _I.device_read(), _P.device_read(), _a.device_read(), c, x, y);
        else if (n_quad <= 12)
            mass_action<12>(n_elem, n_quad, n_basis, _I.device_read(), _P.device_read(), _a.device_read(), c, x, y);
        else if (n_quad <= 16)
            mass_action<16>(n_elem, n_quad, n_basis, _I.device_read(), _P.device_read(), _a.device_read(), c, x, y);
        else if (n_quad <= 24)
            mass_action<24>(n_elem, n_quad, n_basis, _I.device_read(), _P.device_read(), _a.device_read(), c, x, y);
        else if (n_quad <= 32)
            mass_action<32>(n_elem, n_quad, n_basis, _I.device_read(), _P.device_read(), _a.device_read(), c, x, y);
        else if (n_quad <= 64)
            mass_action<64>(n_elem, n_quad, n_basis, _I.device_read(), _P.device_read(), _a.device_read(), c, x, y);
        else
            cuddh_error("MassMatrix error: quadrature rules with more than 64 points not yet supported.");
    }

    void MassMatrix::action(const double * x, double * y) const
    {
        zeros(ndof, y);
        action(1.0, x, y);
    }

    static void init_diag_mass(int n_elem,
                               int n_basis,
                               const double * a,
                               const double * _detJ,
                               const QuadratureRule& quad,
                               const int * _I,
                               double * op)
    {
        auto detJ = reshape(_detJ, n_basis, n_basis, n_elem);
        auto I = reshape(_I, n_basis, n_basis, n_elem);
        
        host_device_dvec _w(n_basis);
        double * h_w = _w.host_write();
        for (int i = 0; i < n_basis; ++i)
            h_w[i] = quad.w(i);

        auto w = reshape(_w.device_read(), n_basis);

        zeros(ndof, op);

        forall_2d(n_basis, n_basis, n_elem, [=] __device__ (int el) -> void {
            const int i = threadIdx.x;
            const int j = threadIdx.y;

            const int idx = I(i, j, el);

            double m = w(i) * w(j) * detJ(i, j, el);
            if (a)
                m *= a[idx];

            AtomicAdd(op + idx, m);
        });

        forall(ndof, [=] __device__ (int i) -> void {
            op[i] = 1.0 / op[i];
        });
    }

    DiagInvMassMatrix::DiagInvMassMatrix(const H1Space& fem)
        : ndof{fem.size()},
          _p(ndof),
          _I{fem.global_indices()}
    {
        const int n_elem = fem.mesh().n_elem();
        const int n_basis = fem.basis().size();

        auto& q = fem.basis().quadrature();
        auto& metrics = fem.mesh().element_metrics(q);
        auto detJ = metrics.measures().device_read();
        
        init_diag_mass(n_elem, n_basis, nullptr, detJ, q, _I.device_read(), _p.device_write());
    }

    DiagInvMassMatrix::DiagInvMassMatrix(const double * a, const H1Space& fem)
        : ndof(fem.size()),
          _p(ndof),
          _I{fem.global_indices()}
    {
        const int n_elem = fem.mesh().n_elem();
        const int n_basis = fem.basis().size();

        auto& q = fem.basis().quadrature();
        auto& metrics = fem.mesh().element_metrics(q);
        auto detJ = metrics.measures().device_read();

        init_diag_mass(n_elem, n_basis, a, detJ, q, _I.device_read(), _p.device_write());
    }

    void DiagInvMassMatrix::action(double c, const double * x, double * y) const
    {
        const double * p = _p.device_read();

        forall(ndof, [=] __device__ (int i) -> void {
            y[i] += c * p[i] * x[i];
        });
    }

    void DiagInvMassMatrix::action(const double * x, double * y) const
    {
        const double * p = _p.device_read();

        forall(ndof, [=] __device__ (int i) -> void {
            y[i] = p[i] * x[i];
        });
    }
} // namespace cuddh
