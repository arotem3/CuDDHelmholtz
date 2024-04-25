#include "MassMatrix.hpp"

namespace cuddh
{
    template <int NQ>
    static void init_mass_matrix(int n_elem,
                                 int n_basis,
                                 int n_quad,
                                 const double * a, /* device */
                                 const double * d_detJ, /* device */
                                 const QuadratureRule& quad,
                                 const int * d_I, /* device */
                                 const double * d_P, /* device */
                                 double * d_op /* device */)
    {
        auto detJ = reshape(d_detJ, n_quad, n_quad, n_elem);
        auto P_ = reshape(d_P, n_quad, n_basis);
        auto I = reshape(d_I, n_basis, n_basis, n_elem);

        auto op = reshape(d_op, n_quad, n_quad, n_elem);

        host_device_dvec _w(n_quad);
        double * h_w = _w.host_write();
        for (int i = 0; i < n_quad; ++i)
            h_w[i] = quad.w(i);
        
        auto w = reshape(_w.device_read(), n_quad);

        forall_2d(n_quad, n_quad, n_elem, [=] __device__ (int el) mutable -> void
        {
            __shared__ double Q[NQ][NQ];
            __shared__ double z[NQ][NQ];
            __shared__ double P[NQ][NQ];

            const int tx = threadIdx.x;
            const int ty = threadIdx.y;

            // copy P
            if (ty < n_basis)
                P[tx][ty] = P_(tx, ty);

            // copy global a to element
            if (tx < n_basis && ty < n_basis)
            {
                const int idx = I(tx, ty, el);
                Q[tx][ty] = (a) ? a[idx] : 1.0;
            }
            __syncthreads();

            // evaluate on quadrature points
            if (ty < n_basis)
            {
                double px = 0.0;
                for (int k = 0; k < n_basis; ++k)
                    px += P[tx][k] * Q[k][ty];
                z[tx][ty] = px;
            }
            __syncthreads();

            double ppx = 0.0;
            for (int l = 0; l < n_basis; ++l)
                ppx += P[ty][l] * z[tx][l];
            ppx *= w(tx) * w(ty) * detJ(tx, ty, el);

            op(tx, ty, el) = ppx;
        });
    }

    MassMatrix::MassMatrix(const H1Space& fem_)
        : fem{fem_},
          ndof{fem.size()},
          n_elem{fem.mesh().n_elem()},
          n_basis{fem.basis().size()},
          n_quad{n_basis + fem.mesh().max_element_order()},
          _P(n_quad * n_basis),
          _a(n_quad * n_quad * n_elem)
    {
        QuadratureRule quad(n_quad, QuadratureRule::GaussLegendre);
        
        const int * d_I = fem.global_indices(MemorySpace::DEVICE);
        auto& metrics = fem.mesh().element_metrics(quad);
        const double * d_detJ = metrics.measures(MemorySpace::DEVICE);
        double * d_op = _a.device_write();

        fem.basis().eval(n_quad, quad.x(), _P.host_write());

        const double * d_P = _P.device_read();

        if (n_quad <= 4)
            init_mass_matrix<4>(n_elem, n_basis, n_quad, nullptr, d_detJ, quad, d_I, d_P, d_op);
        else if (n_quad <= 8)
            init_mass_matrix<8>(n_elem, n_basis, n_quad, nullptr, d_detJ, quad, d_I, d_P, d_op);
        else if (n_quad <= 16)
            init_mass_matrix<16>(n_elem, n_basis, n_quad, nullptr, d_detJ, quad, d_I, d_P, d_op);
        else if (n_quad <= 24)
            init_mass_matrix<24>(n_elem, n_basis, n_quad, nullptr, d_detJ, quad, d_I, d_P, d_op);
        else if (n_quad <= 32)
            init_mass_matrix<32>(n_elem, n_basis, n_quad, nullptr, d_detJ, quad, d_I, d_P, d_op);
        else
            cuddh_error("MassMatrix error: quadrature rules with more than 32 points not yet supported.");
    }

    MassMatrix::MassMatrix(const double * a_, const H1Space& fem_)
        : fem{fem_},
          ndof{fem.size()},
          n_elem{fem.mesh().n_elem()},
          n_basis{fem.basis().size()},
          n_quad(1 + 3*n_basis/2 + fem.mesh().max_element_order()),
          _P(n_quad * n_basis),
          _a(n_quad * n_quad * n_elem)
    {
        QuadratureRule quad(n_quad, QuadratureRule::GaussLegendre);

        const int * d_I = fem.global_indices(MemorySpace::DEVICE);
        auto& metrics = fem.mesh().element_metrics(quad);
        const double * d_detJ = metrics.measures(MemorySpace::DEVICE);
        double * d_op = _a.device_write();

        fem.basis().eval(n_quad, quad.x(), _P.host_write());

        const double * d_P = _P.device_read();

        if (n_quad <= 4)
            init_mass_matrix<4>(n_elem, n_basis, n_quad, a_, d_detJ, quad, d_I, d_P, d_op);
        else if (n_quad <= 8)
            init_mass_matrix<8>(n_elem, n_basis, n_quad, a_, d_detJ, quad, d_I, d_P, d_op);
        else if (n_quad <= 16)
            init_mass_matrix<16>(n_elem, n_basis, n_quad, a_, d_detJ, quad, d_I, d_P, d_op);
        else if (n_quad <= 24)
            init_mass_matrix<24>(n_elem, n_basis, n_quad, a_, d_detJ, quad, d_I, d_P, d_op);
        else if (n_quad <= 32)
            init_mass_matrix<32>(n_elem, n_basis, n_quad, a_, d_detJ, quad, d_I, d_P, d_op);
        else
            cuddh_error("MassMatrix error: quadrature rules with more than 32 points not yet supported.");
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

        forall_2d(n_quad, n_quad, n_elem, [=] __device__ (int el) -> void
        {
            __shared__ double u[NQ][NQ];
            __shared__ double Pu[NQ][NQ];
            __shared__ double P[NQ][NQ];

            const int tx = threadIdx.x;
            const int ty = threadIdx.y;

            int idx;

            // copy P
            if (ty < n_basis)
                P[tx][ty] = P_(tx, ty);

            // copy global dofs to element
            if (tx < n_basis && ty < n_basis)
            {
                idx = I(tx, ty, el);
                u[tx][ty] = x[idx];
            }
            __syncthreads();

            // evaluate on quadrature points
            if (ty < n_basis)
            {
                double pu = 0.0;
                for (int k = 0; k < n_basis; ++k)
                    pu += P[tx][k] * u[k][ty];
                Pu[tx][ty] = pu;
            }
            __syncthreads();

            double ppu = 0.0;
            for (int l = 0; l < n_basis; ++l)
                ppu += P[ty][l] * Pu[tx][l];
            u[tx][ty] = a(tx, ty, el) * ppu; // u(x) * a(x) * w_{ij} * detJ_{ij}
            __syncthreads();

            // integrate
            if (ty < n_basis)
            {
                double qu = 0.0;
                for (int j = 0; j < n_quad; ++j)
                    qu += P[j][ty] * u[tx][j];
                Pu[tx][ty] = qu;
            }
            __syncthreads();

            if (tx < n_basis && ty < n_basis)
            {
                double qqu = 0.0;
                for (int i = 0; i < n_quad; ++i)
                    qqu += P[i][tx] * Pu[i][ty];
                qqu *= c;

                atomicAdd(y + idx, qqu);
            }
        });
    }

    void MassMatrix::action(double c, const double * x, double * y) const
    {
        const int * I = fem.global_indices(MemorySpace::DEVICE);
        const double * a = _a.device_read();
        const double * P = _P.device_read();

        if (n_quad <= 4)
            mass_action<4>(n_elem, n_quad, n_basis, I, P, a, c, x, y);
        else if (n_quad <= 8)
            mass_action<8>(n_elem, n_quad, n_basis, I, P, a, c, x, y);
        else if (n_quad <= 12)
            mass_action<12>(n_elem, n_quad, n_basis, I, P, a, c, x, y);
        else if (n_quad <= 16)
            mass_action<16>(n_elem, n_quad, n_basis, I, P, a, c, x, y);
        else if (n_quad <= 24)
            mass_action<24>(n_elem, n_quad, n_basis, I, P, a, c, x, y);
        else if (n_quad <= 32)
            mass_action<32>(n_elem, n_quad, n_basis, I, P, a, c, x, y);
        else
            cuddh_error("MassMatrix error: quadrature rules with more than 32 points not yet supported.");
    }

    void MassMatrix::action(const double * x, double * y) const
    {
        zeros(ndof, y);
        action(1.0, x, y);
    }

    static void init_diag_mass(int ndof,
                               int n_elem,
                               int n_basis,
                               const double * a,
                               const double * d_detJ,
                               const QuadratureRule& quad,
                               const int * d_I,
                               double * op)
    {
        auto detJ = reshape(d_detJ, n_basis, n_basis, n_elem);
        auto I = reshape(d_I, n_basis, n_basis, n_elem);
        
        host_device_dvec _w(n_basis);
        double * h_w = _w.host_write();
        for (int i = 0; i < n_basis; ++i)
            h_w[i] = quad.w(i);

        auto w = reshape(_w.device_read(), n_basis);

        zeros(ndof, op);

        forall_2d(n_basis, n_basis, n_elem, [=] __device__ (int el) -> void
        {
            const int i = threadIdx.x;
            const int j = threadIdx.y;

            const int idx = I(i, j, el);

            double m = w(i) * w(j) * detJ(i, j, el);
            if (a)
                m *= a[idx];

            atomicAdd(op + idx, m);
        });

        forall(ndof, [=] __device__ (int i) -> void
        {
            op[i] = 1.0 / op[i];
        });
    }

    DiagInvMassMatrix::DiagInvMassMatrix(const H1Space& fem_)
        : fem{fem_},
          ndof{fem.size()},
          _p(ndof)
    {
        const int n_elem = fem.mesh().n_elem();
        const int n_basis = fem.basis().size();

        auto& q = fem.basis().quadrature();
        auto& metrics = fem.mesh().element_metrics(q);
        const double * detJ = metrics.measures(MemorySpace::DEVICE);
        const int * I = fem.global_indices(MemorySpace::DEVICE);
        double * op = _p.device_write();
        
        init_diag_mass(ndof, n_elem, n_basis, nullptr, detJ, q, I, op);
    }

    DiagInvMassMatrix::DiagInvMassMatrix(const double * a, const H1Space& fem_)
        : fem{fem_},
          ndof(fem.size()),
          _p(ndof)
    {
        const int n_elem = fem.mesh().n_elem();
        const int n_basis = fem.basis().size();

        auto& q = fem.basis().quadrature();
        auto& metrics = fem.mesh().element_metrics(q);
        const double * detJ = metrics.measures(MemorySpace::DEVICE);
        const int * I = fem.global_indices(MemorySpace::DEVICE);
        double * op = _p.device_write();

        init_diag_mass(ndof, n_elem, n_basis, a, detJ, q, I, op);
    }

    void DiagInvMassMatrix::action(double c, const double * x, double * y) const
    {
        const double * p = _p.device_read();

        forall(ndof, [=] __device__ (int i) -> void
        {
            y[i] += c * p[i] * x[i];
        });
    }

    void DiagInvMassMatrix::action(const double * x, double * y) const
    {
        const double * p = _p.device_read();

        forall(ndof, [=] __device__ (int i) -> void
        {
            y[i] = p[i] * x[i];
        });
    }
} // namespace cuddh
