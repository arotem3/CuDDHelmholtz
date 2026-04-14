#pragma once

#include <concepts>

#include "Basis.hpp"
#include "FEM3D/H1Space3D.hpp"
#include "HostDeviceArray.hpp"
#include "Mesh3D/Element.hpp"
#include "Projection.hpp"
#include "QuadratureRule.hpp"
#include "Tensor.hpp"
#include "cuddh_config.hpp"
#include "forall.hpp"

namespace cuddh
{
    template <typename F>
    concept Func3D = std::invocable<F, double3>;

    template <typename value_t>
    class GridFunc3D
    {
    public:
        using value_type = value_t;

        explicit GridFunc3D(const H1Space3D &fem) : _fem{fem}
        {
            int nb = fem.basis().size();
            int nel = fem.mesh().n_elem();
            values.resize(nb * nb * nb * nel);
        }

        int n_elem() const { return _fem.mesh().n_elem(); }
        int n_basis() const { return _fem.basis().size(); }
        const H1Space3D &h1_space() const { return _fem; }

        TensorWrapper<4, const value_t> read(MemorySpace m) const
        {
            return reshape(values.read(m), n_basis(), n_basis(), n_basis(), n_elem());
        }

        TensorWrapper<4, value_t> write(MemorySpace m)
        {
            return reshape(values.write(m), n_basis(), n_basis(), n_basis(), n_elem());
        }

        TensorWrapper<4, value_t> read_write(MemorySpace m)
        {
            return reshape(values.read_write(m), n_basis(), n_basis(), n_basis(), n_elem());
        }

        template <std::invocable<value_t> Func>
            requires(std::is_convertible_v<std::invoke_result_t<Func, value_t>, value_t>)
        void apply(Func &&f)
        {
            auto v = read_write(MemorySpace::DEVICE);
            const int n = v.size();
            forall(n, [=] __device__(int i) mutable { v[i] = std::invoke(f, v[i]); });
        }

        template <std::invocable<value_t> Func, typename result_t = std::invoke_result_t<Func, value_t>>
        GridFunc3D<result_t> transform(Func &&f) const
        {
            GridFunc3D<result_t> F(_fem);
            auto fvals = F.write(MemorySpace::DEVICE);
            auto vals = read(MemorySpace::DEVICE);
            forall(vals.size(), [=] __device__(int i) mutable { fvals[i] = std::invoke(f, vals[i]); });
            return F;
        }

    private:
        const H1Space3D &_fem;
        HostDeviceArray<value_t> values;
    };

    /// @brief L2-project f onto the FEM basis using the given quadrature rule.
    template <Func3D Func, typename value_t = std::invoke_result_t<Func, double3>, int MXQ = 8>
    GridFunc3D<value_t> gridfunc(const H1Space3D &fem, const QuadratureRule &quad, Func &&f)
    {
        static_assert(MXQ * MXQ * MXQ <= 1024, "MXQ^3 cannot exceed max threads per block (1024).");

        const int n_quad = quad.size();
        cuddh_verify(n_quad <= MXQ, printf("gridfunc error: quad.size() (=%d) > %d not supported.\n", n_quad, MXQ));

        const int n_basis = fem.basis().size();
        cuddh_verify(n_basis <= MXQ, printf("gridfunc error: basis.size() (=%d) > %d not supported.\n", n_basis, MXQ));

        GridFunc3D<value_t> gf(fem);
        auto F = gf.write(MemorySpace::DEVICE);

        const auto mesh = fem.mesh().to_device();

        HostDeviceArray<double> _p(n_basis * n_quad);
        polynomial_projection_matrix(reshape(_p.write(MemorySpace::HOST), n_quad, n_basis), fem.basis(), quad);

        const auto p = reshape(_p.read(MemorySpace::DEVICE), n_quad, n_basis);
        const auto qx = quad.x(MemorySpace::DEVICE);

        const int nel = gf.n_elem();
        const int n = std::max(n_basis, n_quad);

        forall_3d(n, n, n, nel, [=] __device__(int el) mutable {
            const int i = threadIdx.x;
            const int j = threadIdx.y;
            const int k = threadIdx.z;

            __shared__ value_t Fel[MXQ][MXQ][MXQ];
            __shared__ double P[MXQ][MXQ];

            if (k == 0 && i < n_quad && j < n_basis)
                P[i][j] = p(i, j);

            __shared__ HexElement element;
            if (i == 0 && j == 0 && k == 0)
                element = mesh.element(el);

            __syncthreads();

            if (i < n_quad && j < n_quad && k < n_quad)
            {
                double3 xi{qx[i], qx[j], qx[k]};
                double3 x = element.physical_coordinates(xi);
                Fel[i][j][k] = std::invoke(f, x);
            }
            __syncthreads();

            value_t Px{};
            if (i < n_basis && j < n_quad && k < n_quad)
                for (int l = 0; l < n_quad; ++l)
                    Px = Px + P[l][i] * Fel[l][j][k];
            __syncthreads();
            if (i < n_quad && j < n_quad && k < n_quad)
                Fel[i][j][k] = Px;
            __syncthreads();

            value_t Py{};
            if (i < n_basis && j < n_basis && k < n_quad)
                for (int l = 0; l < n_quad; ++l)
                    Py = Py + P[l][j] * Fel[i][l][k];
            __syncthreads();
            if (i < n_quad && j < n_quad && k < n_quad)
                Fel[i][j][k] = Py;
            __syncthreads();

            if (i < n_basis && j < n_basis && k < n_basis)
            {
                value_t Pz{};
                for (int l = 0; l < n_quad; ++l)
                    Pz = Pz + P[l][k] * Fel[i][j][l];
                F(i, j, k, el) = Pz;
            }
        });

        return gf;
    }

    /// @brief L2-project f onto the FEM basis using Gauss-Legendre quadrature.
    ///
    /// Gauss-Legendre nodes lie strictly inside each element, so discontinuous
    /// coefficients are never evaluated on element boundaries.
    template <Func3D Func, typename value_t = std::invoke_result_t<Func, double3>>
    GridFunc3D<value_t> gridfunc(const H1Space3D &fem, Func &&f)
    {
        QuadratureRule quad(fem.basis().size(), QuadratureRule::GaussLegendre);
        return gridfunc<Func, value_t>(fem, quad, std::forward<Func>(f));
    }
} // namespace cuddh
