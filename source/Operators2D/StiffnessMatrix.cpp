#include "Operators2D/StiffnessMatrix.hpp"

#include "linalg.hpp"

namespace cuddh
{
    static HostDeviceArray<dsym2x2> setup_geometric_factors(const Mesh2D &mesh, const QuadratureRule &quad)
    {
        const int n_elem = mesh.n_elem();
        const int n = quad.size();

        auto w = quad.w(MemorySpace::DEVICE);
        auto x = quad.x(MemorySpace::DEVICE);

        auto d_mesh = mesh.to_device();

        HostDeviceArray<dsym2x2> g(n * n * n_elem);
        auto G = reshape(g.device_write(), n, n, n_elem);

        forall_2d(n, n, n_elem, [=] __device__(int el) mutable -> void {
            const auto [i, j, _] = threadIdx;

            __shared__ QuadElement element;
            if (i == 0 && j == 0)
                element = d_mesh.element(el);
            __syncthreads();

            const double2x2 J = element.jacobian({x(i), x(j)});
            const double W = w(i) * w(j) / det(J);

            dsym2x2 gij;
            gij(0, 0) = W * (J(1, 1) * J(1, 1) + J(0, 1) * J(0, 1));
            gij(1, 0) = -W * (J(1, 0) * J(1, 1) + J(0, 0) * J(0, 1));
            gij(1, 1) = W * (J(1, 0) * J(1, 0) + J(0, 0) * J(0, 0));

            G(i, j, el) = gij;
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
        basis.deriv(n_basis, quad.x(MemorySpace::HOST), _D.write(MemorySpace::HOST));

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

    // Assemble the 2D element stiffness matrix using the explicit formula:
    //   K[(a,b),(c,d)] = δ(b,d)*Σ_k D(k,a)*G(k,b)_00*D(k,c)
    //                  + D(c,a)*G(c,b)_01*D(b,d)
    //                  + D(a,c)*G(a,d)_01*D(d,b)
    //                  + δ(a,c)*Σ_l D(l,b)*G(a,l)_11*D(l,d)
    // with D(i,j) = D_ptr[i + nb*j] and G(k,l,el) at G_ptr[k + nb*l + nb^2*el].
    bool StiffnessMatrix::assemble(double c, SparseMatrix<double> &S_out) const
    {
        const int nb = n_basis;
        auto D = reshape(_D.host_read(), nb, nb);
        auto G = reshape(_G.host_read(), nb, nb, n_elem);
        auto I = fem.global_indices(MemorySpace::HOST);

        for (int el = 0; el < n_elem; ++el)
            for (int a = 0; a < nb; ++a)
                for (int b = 0; b < nb; ++b)
                {
                    const int row = I(a, b, el);
                    for (int cv = 0; cv < nb; ++cv)
                        for (int d = 0; d < nb; ++d)
                        {
                            const int col = I(cv, d, el);
                            double val = 0.0;
                            if (b == d)
                                for (int k = 0; k < nb; ++k)
                                    val += D(k, a) * G(k, b, el)(0, 0) * D(k, cv);
                            val += D(cv, a) * G(cv, b, el)(0, 1) * D(b, d);
                            val += D(a, cv) * G(a, d, el)(0, 1) * D(d, b);
                            if (a == cv)
                                for (int l = 0; l < nb; ++l)
                                    val += D(l, b) * G(a, l, el)(1, 1) * D(l, d);
                            S_out.set_value(row, col, c * val);
                        }
                }
        return true;
    }

    bool StiffnessMatrix::assemble(std::complex<double> c, SparseMatrix<double, true> &S_out) const
    {
        const int nb = n_basis;
        auto D = reshape(_D.host_read(), nb, nb);
        auto G = reshape(_G.host_read(), nb, nb, n_elem);
        auto I = fem.global_indices(MemorySpace::HOST);

        for (int el = 0; el < n_elem; ++el)
            for (int a = 0; a < nb; ++a)
                for (int b = 0; b < nb; ++b)
                {
                    const int row = I(a, b, el);
                    for (int cv = 0; cv < nb; ++cv)
                        for (int d = 0; d < nb; ++d)
                        {
                            const int col = I(cv, d, el);
                            double val = 0.0;
                            if (b == d)
                                for (int k = 0; k < nb; ++k)
                                    val += D(k, a) * G(k, b, el)(0, 0) * D(k, cv);
                            val += D(cv, a) * G(cv, b, el)(0, 1) * D(b, d);
                            val += D(a, cv) * G(a, d, el)(0, 1) * D(d, b);
                            if (a == cv)
                                for (int l = 0; l < nb; ++l)
                                    val += D(l, b) * G(a, l, el)(1, 1) * D(l, d);
                            S_out.set_value(row, col, c * val);
                        }
                }
        return true;
    }
} // namespace cuddh
