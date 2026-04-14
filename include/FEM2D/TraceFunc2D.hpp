#pragma once

#include "FEM2D/GridFunc2D.hpp"
#include "FEM2D/H1Space2D.hpp"
#include "HostDeviceArray.hpp"
#include "Projection.hpp"
#include "QuadratureRule.hpp"
#include "Tensor.hpp"
#include "cuddh_config.hpp"

namespace cuddh
{
    template <typename value_t>
    class TraceFunc2D
    {
    public:
        using value_type = value_t;

        explicit TraceFunc2D(const TraceSpace2D &tr) : tr{tr}
        {
            int nb = tr.h1_space().basis().size();
            int nf = tr.n_faces();

            values.resize(nb * nf);
        }

        int n_faces() const { return tr.n_faces(); }
        int n_basis() const { return tr.h1_space().basis().size(); }
        const TraceSpace2D &trace_space() const { return tr; }

        MatrixWrapper<const value_t> read(MemorySpace m) const { return reshape(values.read(m), n_basis(), n_faces()); }
        MatrixWrapper<value_t> write(MemorySpace m) { return reshape(values.write(m), n_basis(), n_faces()); }
        MatrixWrapper<value_t> read_write(MemorySpace m) { return reshape(values.read_write(m), n_basis(), n_faces()); }

    private:
        const TraceSpace2D &tr;
        HostDeviceArray<value_t> values;
    };

    template <typename value_t>
    TraceFunc2D<value_t> trace(const TraceSpace2D &tr, const GridFunc2D<value_t> &gf)
    {
        const int nb = gf.n_basis();

        TraceFunc2D<value_t> tf(tr);
        auto T = tf.write(MemorySpace::DEVICE);

        auto F = gf.read(MemorySpace::DEVICE);

        const auto &mesh = tr.h1_space().mesh().to_device();
        const auto faces = tr.faces(MemorySpace::DEVICE);

        forall_1d(nb, faces.size(), [=] __device__(int f) mutable {
            const auto i = threadIdx.x;

            __shared__ EdgeConnectivity connectivity;
            if (i == 0)
                connectivity = mesh.edge_connectivity(faces[f]);
            __syncthreads();

            const int el = connectivity.elements[0];

            const auto [x, y] = edge2vol(nb, i, connectivity.labels[0]);
            T(i, f) = F(x, y, el);
        });

        return tf;
    }

    /// @brief L2-project a boundary function onto the trace space.
    ///
    /// For each boundary face the function is evaluated at n_quad quadrature
    /// points, then the 1-D polynomial projection matrix is applied so that
    /// the result lives in the span of the n_basis nodal basis functions on
    /// that face.
    template <typename Func, typename value_t = std::invoke_result_t<Func, double2>, int MXQ = 8>
    TraceFunc2D<value_t> trace(const TraceSpace2D &tr, Func &&f)
    {
        const int n_basis = tr.h1_space().basis().size();
        const int n_faces = tr.n_faces();

        QuadratureRule quad(n_basis, QuadratureRule::GaussLegendre);
        const int n_quad = quad.size();

        HostDeviceArray<double> _p(n_quad * n_basis);
        polynomial_projection_matrix(reshape(_p.write(MemorySpace::HOST), n_quad, n_basis), tr.h1_space().basis(),
                                     quad);
        const auto p = reshape(_p.read(MemorySpace::DEVICE), n_quad, n_basis);
        const auto qx = quad.x(MemorySpace::DEVICE);

        const auto d_mesh = tr.h1_space().mesh().to_device();
        const auto faces = tr.faces(MemorySpace::DEVICE);

        TraceFunc2D<value_t> tf(tr);
        auto T = tf.write(MemorySpace::DEVICE);

        const int n = std::max(n_basis, n_quad);
        forall_1d(n, n_faces, [=] __device__(int face) mutable {
            const int k = threadIdx.x;

            __shared__ value_t Ff[MXQ];
            __shared__ double P[MXQ][MXQ]; // P[quad][basis]

            if (k < n_quad)
            {
                for (int i = 0; i < n_basis; ++i)
                    P[k][i] = p(k, i);

                const Edge edge = d_mesh.edge(faces[face]);
                Ff[k] = std::invoke(f, edge.physical_coordinates(qx[k]));
            }
            __syncthreads();

            if (k < n_basis)
            {
                value_t Tf{};
                for (int q = 0; q < n_quad; ++q)
                    Tf = Tf + P[q][k] * Ff[q];
                T(k, face) = Tf;
            }
        });

        return tf;
    }
} // namespace cuddh
