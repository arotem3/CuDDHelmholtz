#include "Operators2D/StiffnessMatrix.hpp"

namespace cuddh
{
    static HostDeviceArray<dsym2x2> setup_geometric_factors(const Mesh2D &mesh, const QuadratureRule &quad)
    {
        const int n_elem = mesh.n_elem();
        const int n = quad.size();

        thrust::host_vector<double> h_w(n);
        for (int i = 0; i < n; ++i)
            h_w[i] = quad.w(i);
        thrust::device_vector<double> d_w = h_w;
        auto w = reshape(thrust::raw_pointer_cast(d_w.data()), n);

        const double *_J = mesh.element_metrics(quad).jacobians(MemorySpace::DEVICE);
        auto J = reshape(_J, 2, 2, n, n, n_elem);

        HostDeviceArray<dsym2x2> g(n * n * n_elem);
        auto G = reshape(g.device_write(), n, n, n_elem);

        forall_2d(n, n, n_elem, [=] __device__(int el) mutable -> void {
            const auto [i, j, _] = threadIdx;

            const double W = w(i) * w(j);
            const double Y_eta = J(1, 1, i, j, el);
            const double X_eta = J(0, 1, i, j, el);
            const double Y_xi = J(1, 0, i, j, el);
            const double X_xi = J(0, 0, i, j, el);

            const double detJ = X_xi * Y_eta - X_eta * Y_xi;

            dsym2x2 g_el;
            g_el(0, 0) = W * (Y_eta * Y_eta + X_eta * X_eta) / detJ;
            g_el(1, 0) = -W * (Y_xi * Y_eta + X_xi * X_eta) / detJ;
            g_el(1, 1) = W * (Y_xi * Y_xi + X_xi * X_xi) / detJ;

            G(i, j, el) = g_el;
        });

        return g;
    }

    StiffnessMatrix::StiffnessMatrix(const H1Space2D &fem_)
        : Operator<double>(fem_.size()),
          fem{fem_},
          n_elem{fem.mesh().n_elem()},
          n_basis{fem.basis().size()},
          _D(n_basis * n_basis)
    {
        const auto &basis = fem.basis();
        const auto &quad = basis.quadrature();
        basis.deriv(n_basis, quad.x(), _D.host_write());

        _G = setup_geometric_factors(fem.mesh(), quad);
    }

    template <int NB>
    static void stiffness_action(int n_elem, int n_basis, MatrixWrapper<const double> D,
                                 TensorWrapper<3, const dsym2x2> G, TensorWrapper<3, const int> I, double c,
                                 const double *d_u, double *d_out)
    {
        cuddh_verify(NB >= n_basis,
                     printf("stiffness error: stiffness kernel called with NB = %d < n_basis = %d\n", NB, n_basis));

        forall_2d(n_basis, n_basis, n_elem, [=] __device__(int el) {
            __shared__ double s_u[NB][NB];
            __shared__ double2 s_grad[NB][NB];
            __shared__ double s_D[NB][NB];

            const auto [k, l, _] = threadIdx;

            int idx = I(k, l, el);
            s_u[k][l] = d_u[idx];
            s_D[k][l] = D(k, l);
            __syncthreads();

            double2 tgrad{0.0, 0.0};

#pragma unroll NB
            for (int i = 0; i < n_basis; ++i)
                tgrad.x += s_D[k][i] * s_u[i][l];

#pragma unroll NB
            for (int i = 0; i < n_basis; ++i)
                tgrad.y += s_D[l][i] * s_u[k][i];

            s_grad[k][l] = G(k, l, el) * tgrad;
            __syncthreads();

            double Au = 0.0;

#pragma unroll NB
            for (int i = 0; i < n_basis; ++i)
                Au += s_D[i][k] * s_grad[i][l].x;

#pragma unroll NB
            for (int i = 0; i < n_basis; ++i)
                Au += s_D[i][l] * s_grad[k][i].y;

            Au *= c;

            atomicAdd(d_out + idx, Au);
        });
    }

    void StiffnessMatrix::action(double c, const double *x, double *y) const
    {
        auto D = reshape(_D.device_read(), n_basis, n_basis);
        auto G = reshape(_G.device_read(), n_basis, n_basis, n_elem);
        auto I = reshape(fem.global_indices(MemorySpace::DEVICE), n_basis, n_basis, n_elem);

        if (n_basis <= 4)
            stiffness_action<4>(n_elem, n_basis, D, G, I, c, x, y);
        else if (n_basis <= 8)
            stiffness_action<8>(n_elem, n_basis, D, G, I, c, x, y);
        else if (n_basis <= 12)
            stiffness_action<12>(n_elem, n_basis, D, G, I, c, x, y);
        else if (n_basis <= 16)
            stiffness_action<16>(n_elem, n_basis, D, G, I, c, x, y);
        else if (n_basis <= 24)
            stiffness_action<24>(n_elem, n_basis, D, G, I, c, x, y);
        else
            cuddh_verify(false,
                         printf("StiffnessMatrix::action does not support quadrature rules with more than 24 points."));
    }

    void StiffnessMatrix::action(const double *x, double *y) const
    {
        dla::zeros(this->ndof(), y);
        action(1.0, x, y);
    }
} // namespace cuddh
