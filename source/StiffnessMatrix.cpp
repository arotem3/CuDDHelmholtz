#include "StiffnessMatrix.hpp"

namespace cuddh
{
    static void setup_geometric_factors(int n_elem,
                                    const cuddh::QuadratureRule& quad,
                                    const double * _J,
                                    double * _G)
    {
        const int n_quad = quad.size();

        host_device_dvec _w(n_quad);
        double * h_w = _w.host_write();
        for (int i = 0; i < n_quad; ++i)
            h_w[i] = quad.w(i);
        auto w = reshape(_w.device_read(), n_quad);

        auto J = reshape(_J, 2, 2, n_quad, n_quad, n_elem);
        auto G = reshape(_G, 3, n_quad, n_quad, n_elem);

        forall_2d(n_quad, n_quad, n_elem, [=] __device__ (int el) mutable -> void
        {
            const int i = threadIdx.x;
            const int j = threadIdx.y;

            const double W = w(i) * w(j);
            const double Y_eta = J(1, 1, i, j, el);
            const double X_eta = J(0, 1, i, j, el);
            const double Y_xi  = J(1, 0, i, j, el);
            const double X_xi  = J(0, 0, i, j, el);

            const double detJ = X_xi * Y_eta - X_eta * Y_xi;

            G(0, i, j, el) =  W * (Y_eta * Y_eta + X_eta * X_eta) / detJ;
            G(1, i, j, el) = -W * (Y_xi  * Y_eta + X_xi  * X_eta) / detJ;
            G(2, i, j, el) =  W * (Y_xi  * Y_xi  + X_xi  * X_xi)  / detJ;
        });
    }

    StiffnessMatrix::StiffnessMatrix(const H1Space& fem_)
        : fem{fem_},
          ndof{fem.size()},
          n_elem{fem.mesh().n_elem()},
          n_basis{fem.basis().size()},
          n_quad{fem.mesh().max_element_order() + n_basis},
          _P(n_quad * n_basis),
          _D(n_quad * n_basis),
          _G(3 * n_quad * n_quad * n_elem)
    {
        QuadratureRule quad(n_quad, QuadratureRule::GaussLegendre);
        fem.basis().eval(n_quad, quad.x(), _P.host_write());
        fem.basis().deriv(n_quad, quad.x(), _D.host_write());

        auto& metrics = fem.mesh().element_metrics(quad);
        const double * J = metrics.jacobians(MemorySpace::DEVICE);
        
        double * G = _G.device_write();

        setup_geometric_factors(n_elem, quad, J, G);
    }

    StiffnessMatrix::StiffnessMatrix(const H1Space& fem_, const QuadratureRule& quad)
        : fem{fem_},
          ndof{fem.size()},
          n_elem{fem.mesh().n_elem()},
          n_basis{fem.basis().size()},
          n_quad{quad.size()},
          _P(n_quad * n_basis),
          _D(n_quad * n_basis),
          _G(3 * n_quad * n_quad * n_elem)
    {
        fem.basis().eval(n_quad, quad.x(), _P.host_write());
        fem.basis().deriv(n_quad, quad.x(), _D.host_write());

        auto& metrics = fem.mesh().element_metrics(quad);
        const double * J = metrics.jacobians(MemorySpace::DEVICE);

        double * G = _G.device_write();

        setup_geometric_factors(n_elem, quad, J, G);
    }

    template <int NQ>
    static void stiffness_action(int n_elem,
                                 int n_quad,
                                 int n_basis,
                                 const double * d_P,
                                 const double * d_D,
                                 const double * d_G,
                                 const int * d_I,
                                 double c,
                                 const double * d_u,
                                 double * d_out)
    {
        auto _P = reshape(d_P, n_quad, n_basis);
        auto _D = reshape(d_D, n_quad, n_basis);
        
        auto I = reshape(d_I, n_basis, n_basis, n_elem);
        
        auto G = reshape(d_G, 3, n_quad, n_quad, n_elem);
        
        forall_2d(n_quad, n_quad, n_elem, [=] __device__ (int el) -> void
        {
            __shared__ double u[NQ][NQ];
            __shared__ double Pu[NQ][NQ];
            __shared__ double Du[NQ][NQ];
            __shared__ double F[NQ][NQ][2];
            __shared__ double P[NQ][NQ];
            __shared__ double D[NQ][NQ];

            const int tx = threadIdx.x;
            const int ty = threadIdx.y;

            int idx;

            // copy P and D
            if (ty < n_basis)
            {
                P[tx][ty] = _P(tx, ty);
                D[tx][ty] = _D(tx, ty);
            }

            // copy global dofs to element
            if (tx < n_basis && ty < n_basis)
            {
                idx = I(tx, ty, el);
                u[tx][ty] = d_u[idx];
            }
            __syncthreads();

            // evaluate & differentiate on quadrature points
            if (ty < n_basis)
            {
                double pxu = 0.0, dxu = 0.0;
                for (int k = 0; k < n_basis; ++k)
                {
                    const double uk = u[k][ty];
                    pxu += P[tx][k] * uk;
                    dxu += D[tx][k] * uk;
                }
                Pu[tx][ty] = pxu;
                Du[tx][ty] = dxu;
            }
            __syncthreads();

            const double A = G(0, tx, ty, el);
            const double B = G(1, tx, ty, el);
            const double C = G(2, tx, ty, el);

            double Dx = 0.0, Dy = 0.0;
            for (int l = 0; l < n_basis; ++l)
            {
                Dx += P[ty][l] * Du[tx][l];
                Dy += D[ty][l] * Pu[tx][l];
            }
            F[tx][ty][0] = A * Dx + B * Dy;
            F[tx][ty][1] = B * Dx + C * Dy;
            __syncthreads();

            // integrate
            if (tx < n_basis)
            {
                double df = 0.0, pg = 0.0;
                for (int i = 0; i < n_quad; ++i)
                {
                    df += D[i][tx] * F[i][ty][0];
                    pg += P[i][tx] * F[i][ty][1];
                }
                Du[tx][ty] = df;
                Pu[tx][ty] = pg;
            }
            __syncthreads();

            if (tx < n_basis && ty < n_basis)
            {
                double Su = 0.0;
                for (int j = 0; j < n_quad; ++j)
                    Su += P[j][ty] * Du[tx][j] + D[j][ty] * Pu[tx][j];
                Su *= c;

                atomicAdd(d_out+idx, Su);
            }
        });
    }

    void StiffnessMatrix::action(double c, const double * x, double * y) const
    {
        const double * d_P = _P.device_read();
        const double * d_D = _D.device_read();
        const double * d_G = _G.device_read();
        const int * d_I = fem.global_indices(MemorySpace::DEVICE);

        if (n_quad <= 4)
            stiffness_action<4>(n_elem, n_quad, n_basis, d_P, d_D, d_G, d_I, c, x, y);
        else if (n_quad <= 8)
            stiffness_action<8>(n_elem, n_quad, n_basis, d_P, d_D, d_G, d_I, c, x, y);
        else if (n_quad <= 12)
            stiffness_action<12>(n_elem, n_quad, n_basis, d_P, d_D, d_G, d_I, c, x, y);
        else if (n_quad <= 16)
            stiffness_action<16>(n_elem, n_quad, n_basis, d_P, d_D, d_G, d_I, c, x, y);
        else if (n_quad <= 24)
            stiffness_action<24>(n_elem, n_quad, n_basis, d_P, d_D, d_G, d_I, c, x, y);
        else
            cuddh_error("StiffnessMatrix::action does not support quadrature rules with more than 24 points.");
    }

    void StiffnessMatrix::action(const double * x, double * y) const
    {
        zeros(ndof, y);
        action(1.0, x, y);
    }
} // namespace cuddh
