#pragma once

#include <concepts>

#include "FEM2D/H1Space2D.hpp"
#include "HostDeviceArray.hpp"
#include "Tensor.hpp"
#include "cuddh_config.hpp"

namespace cuddh
{
    template <typename F>
    concept Func2D = std::invocable<F, double2>;

    template <typename value_t>
    class GridFunc2D
    {
    public:
        using value_type = value_t;

        explicit GridFunc2D(const H1Space2D &fem) : fem{fem}
        {
            int nb = fem.basis().size();
            int nel = fem.mesh().n_elem();

            values.resize(nb * nb * nel);
        }

        int n_elem() const { return fem.mesh().n_elem(); }
        int n_basis() const { return fem.basis().size(); }
        const H1Space2D &h1_space() const { return fem; }

        TensorWrapper<3, const value_t> read(MemorySpace m) const
        {
            return reshape(values.read(m), n_basis(), n_basis(), n_elem());
        }

        TensorWrapper<3, value_t> write(MemorySpace m)
        {
            return reshape(values.write(m), n_basis(), n_basis(), n_elem());
        }

        TensorWrapper<3, value_t> read_write(MemorySpace m)
        {
            return reshape(values.read_write(m), n_basis(), n_basis(), n_elem());
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
        GridFunc2D<result_t> transform(Func &&f) const
        {
            GridFunc2D<result_t> F(fem);
            auto fvals = F.write(MemorySpace::DEVICE);
            auto vals = read(MemorySpace::DEVICE);

            forall(vals.size(), [=] __device__(int i) mutable { fvals[i] = std::invoke(f, vals[i]); });

            return F;
        }

    private:
        const H1Space2D &fem;
        HostDeviceArray<value_t> values;
    };

    template <Func2D Func, typename value_t = std::invoke_result_t<Func, double2>, int MXQ = 8>
    GridFunc2D<value_t> gridfunc(const H1Space2D &fem, const QuadratureRule &quad, Func &&f)
    {
        static_assert(MXQ * MXQ <= 1024, "MXQ^2 cannot exceed max threads per block (1024).");

        const int n_quad = quad.size();
        cuddh_verify(n_quad <= MXQ, printf("gridfunc error: quad.size() (=%d) > %d not supported.\n", n_quad, MXQ));

        const int n_basis = fem.basis().size();
        cuddh_verify(n_basis <= MXQ, printf("gridfunc error: basis.size() (=%d) > %d not supported.\n", n_basis, MXQ));

        GridFunc2D<value_t> gf(fem);
        auto F = gf.write(MemorySpace::DEVICE);

        const auto mesh = fem.mesh().to_device();

        HostDeviceArray<value_t> _p(n_basis * n_quad);
        polynomial_projection_matrix(reshape(_p.write(MemorySpace::HOST), n_quad, n_basis), fem.basis(), quad);
        const auto p = reshape(_p.read(MemorySpace::DEVICE), n_quad, n_basis);

        const auto qx = quad.x(MemorySpace::DEVICE);

        const int nel = gf.n_elem();
        const int n = std::max(n_basis, n_quad);

        forall_2d(n, n, nel, [=] __device__(int el) mutable {
            const auto [i, j, _] = threadIdx;

            __shared__ value_t Fel[MXQ][MXQ];
            __shared__ double P[MXQ][MXQ];

            if (i < n_quad && j < n_basis)
                P[i][j] = p(i, j);

            __shared__ QuadElement element;
            if (i == 0 && j == 0)
                element = mesh.element(el);

            __syncthreads();

            if (i < n_quad && j < n_quad)
            {
                double2 xi{qx(i), qx(j)};
                double2 x = element.physical_coordinates(xi);

                Fel[i][j] = std::invoke(f, x);
            }
            __syncthreads();

            value_t PxF{};
            if (i < n_basis && j < n_quad)
                for (int l = 0; l < n_quad; ++l)
                    PxF = PxF + P[l][i] * Fel[l][j];
            __syncthreads();

            if (i < n_quad && j < n_quad)
                Fel[i][j] = PxF;
            __syncthreads();

            if (i < n_basis && j < n_basis)
            {
                value_t PyPxF{};
                for (int l = 0; l < n_quad; ++l)
                    PyPxF = PyPxF + P[l][j] * Fel[i][l];
                F(i, j, el) = PyPxF;
            }
        });

        return gf;
    }

    template <Func2D Func, typename value_t = std::invoke_result_t<Func, double2>>
    GridFunc2D<value_t> gridfunc(const H1Space2D &fem, Func &&f)
    {
        QuadratureRule quad(fem.basis().size(), QuadratureRule::GaussLegendre);
        return gridfunc<Func, value_t>(fem, quad, std::forward<Func>(f));
    }
} // namespace cuddh
