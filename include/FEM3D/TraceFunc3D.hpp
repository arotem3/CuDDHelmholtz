#pragma once

#include "FEM3D/GridFunc3D.hpp"
#include "FEM3D/H1Space3D.hpp"
#include "Mesh3D/Connectivity.hpp"
#include "Mesh3D/Mesh3D.hpp"
#include "Projection.hpp"
#include "QuadratureRule.hpp"
#include "Tensor.hpp"
#include "cuddh_config.hpp"
#include "forall.hpp"

namespace cuddh
{
    template <typename value_t>
    class TraceFunc3D
    {
    public:
        using value_type = value_t;

        explicit TraceFunc3D(const TraceSpace3D &tr) : _tr{tr}
        {
            int nb = tr.h1_space().basis().size();
            int nf = tr.n_faces();
            values.resize(nb * nb * nf);
        }

        int n_faces() const { return _tr.n_faces(); }
        int n_basis() const { return _tr.h1_space().basis().size(); }
        const TraceSpace3D &trace_space() const { return _tr; }

        TensorWrapper<3, const value_t> read(MemorySpace m) const
        {
            return reshape(values.read(m), n_basis(), n_basis(), n_faces());
        }

        TensorWrapper<3, value_t> write(MemorySpace m)
        {
            return reshape(values.write(m), n_basis(), n_basis(), n_faces());
        }

        TensorWrapper<3, value_t> read_write(MemorySpace m)
        {
            return reshape(values.read_write(m), n_basis(), n_basis(), n_faces());
        }

    private:
        const TraceSpace3D &_tr;
        HostDeviceArray<value_t> values;
    };

    /// @brief Extract values of gf onto boundary faces, returning a TraceFunc3D.
    /// tf(i, j, f) = gf(x, y, z, el) where (x, y, z) are the volume basis indices
    /// corresponding to face basis indices (i, j) on face f.
    template <typename value_t>
    TraceFunc3D<value_t> trace(const TraceSpace3D &tr, const GridFunc3D<value_t> &gf)
    {
        const int nb = gf.n_basis();
        const int n_faces = tr.n_faces();

        const auto d_mesh = tr.h1_space().mesh().to_device();
        const auto faces_d = tr.faces(MemorySpace::DEVICE);

        TraceFunc3D<value_t> tf(tr);
        auto T = tf.write(MemorySpace::DEVICE);
        auto F = gf.read(MemorySpace::DEVICE);

        forall_2d(nb, nb, n_faces, [=] __device__(int f) mutable {
            const int i = threadIdx.x;
            const int j = threadIdx.y;
            const FaceConnectivity fc = d_mesh.face_connectivity(faces_d[f]);
            const auto [x, y, z] = face2vol(nb, i, j, fc.label[0]);
            T(i, j, f) = F(x, y, z, fc.elements[0]);
        });

        return tf;
    }

    /// @brief L2-project a boundary function onto the trace space.
    ///
    /// For each boundary face the function is evaluated at a n_quad x n_quad
    /// grid of quadrature points, then the separable 2-D polynomial projection
    /// matrix is applied so that the result lives in the span of the
    /// n_basis x n_basis nodal basis functions on that face.
    template <typename Func, typename value_t = std::invoke_result_t<Func, double3>, int MXQ = 8>
    TraceFunc3D<value_t> trace(const TraceSpace3D &tr, Func &&f)
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

        TraceFunc3D<value_t> tf(tr);
        auto T = tf.write(MemorySpace::DEVICE);

        const int n = std::max(n_basis, n_quad);
        forall_2d(n, n, n_faces, [=] __device__(int face) mutable {
            const int i = threadIdx.x;
            const int j = threadIdx.y;

            __shared__ value_t Ff[MXQ][MXQ]; // evaluated at (n_quad x n_quad) quadrature points
            __shared__ double P[MXQ][MXQ];   // P[quad][basis]

            __shared__ QuadFace qface;
            if (i == 0 && j == 0)
                qface = d_mesh.face(faces[face]);

            if (i < n_quad && j == 0)
                for (int b = 0; b < n_basis; ++b)
                    P[i][b] = p(i, b);

            __syncthreads();

            // evaluate f at quadrature points
            if (i < n_quad && j < n_quad)
                Ff[i][j] = std::invoke(f, qface.physical_coordinates({qx[i], qx[j]}));
            __syncthreads();

            // project in i-direction: PxF[i][j] = sum_q P[q][i] * Ff[q][j]
            value_t PxF{};
            if (i < n_basis && j < n_quad)
                for (int q = 0; q < n_quad; ++q)
                    PxF = PxF + P[q][i] * Ff[q][j];
            __syncthreads();

            if (i < n_quad && j < n_quad)
                Ff[i][j] = PxF;
            __syncthreads();

            // project in j-direction: T[i][j] = sum_q P[q][j] * PxF[i][q]
            if (i < n_basis && j < n_basis)
            {
                value_t Tf{};
                for (int q = 0; q < n_quad; ++q)
                    Tf = Tf + P[q][j] * Ff[i][q];
                T(i, j, face) = Tf;
            }
        });

        return tf;
    }
} // namespace cuddh
