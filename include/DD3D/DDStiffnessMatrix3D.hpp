#ifndef DDH_DD_STIFFNESS_MATRIX_3D_HPP
#define DDH_DD_STIFFNESS_MATRIX_3D_HPP

#include <thrust/device_vector.h>
#include <thrust/universal_vector.h>

#include "EnsembleSpace3D.hpp"
#include "SmallMatrix.hpp"
#include "cuddh_config.hpp"
#include "cuddh_error.hpp"
#include "forall.hpp"

namespace cuddh
{
    template <typename scalar_t, int NB, int MX_NEL>
    struct SubdomainStiffnessMatrix
    {
        using vec_t = cuddh::scalar3<scalar_t>;
        using mat_t = SmallSymmetricMatrix<scalar_t, 3>;

        struct SharedResources
        {
            scalar_t D[NB][NB];
            scalar_t u[MX_NEL * NB * NB * NB];
            vec_t grad[MX_NEL][NB][NB][NB];
        };

        SharedResources &smem;
        mat_t geom;
        int I[3][NB];

        __device__ static constexpr int3 get_index3d(int i)
        {
            int3 idx;
            idx.z = i / (NB * NB);
            idx.y = (i % (NB * NB)) / NB;
            idx.x = (i % (NB * NB)) % NB;

            return idx;
        }

        __device__ SubdomainStiffnessMatrix(SharedResources &mem, int subsp, int nel,
                                            const MatrixWrapper<const scalar_t> &D,
                                            const TensorWrapper<5, const mat_t> &G,
                                            const TensorWrapper<5, const int> &sI)
            : smem{mem}
        {
            const auto [x, y, z] = get_index3d(threadIdx.x);
            const auto el = threadIdx.y;

            if (el == 0 && z == 0)
                smem.D[x][y] = D(x, y);

            for (int i = 0; i < NB; ++i)
            {
                I[0][i] = (el < nel) ? sI(i, y, z, el, subsp) : -1;
                I[1][i] = (el < nel) ? sI(x, i, z, el, subsp) : -1;
                I[2][i] = (el < nel) ? sI(x, y, i, el, subsp) : -1;
            }

            geom = (el < nel) ? G(x, y, z, el, subsp) : mat_t{};

            __syncthreads();
        }

        __device__ scalar_t operator()(scalar_t in) const
        {
            const auto [x, y, z] = get_index3d(threadIdx.x);
            const auto el = threadIdx.y;
            const int tid = threadIdx.x + blockDim.x * threadIdx.y;

            smem.u[tid] = in;
            __syncthreads();

            vec_t grad{0, 0, 0};

            if (I[0][0] >= 0) // el < nel
            {
                for (int i = 0; i < NB; ++i)
                {
                    grad.x += smem.D[x][i] * smem.u[I[0][i]];
                    grad.y += smem.D[y][i] * smem.u[I[1][i]];
                    grad.z += smem.D[z][i] * smem.u[I[2][i]];
                }
            }
            __syncthreads();

            smem.u[tid] = 0;
            smem.grad[el][x][y][z] = geom * grad;
            __syncthreads();

            scalar_t Su = 0;

            for (int i = 0; i < NB; ++i)
            {
                Su += smem.D[i][x] * smem.grad[el][i][y][z].x + smem.D[i][y] * smem.grad[el][x][i][z].y +
                      smem.D[i][z] * smem.grad[el][x][y][i].z;
            }
            atomicAdd(smem.u + I[0][x], Su);
            __syncthreads();

            return smem.u[tid];
        }
    };

    template <typename scalar_t>
    struct DeviceDDStiffnessMatrix3D
    {
        MatrixWrapper<const scalar_t> D;
        TensorWrapper<5, const SmallSymmetricMatrix<scalar_t, 3>> G;
        TensorWrapper<5, const int> I;

        template <int NB, int MX_NEL>
        __device__ SubdomainStiffnessMatrix<scalar_t, NB, MX_NEL> subspace_op(
            int subsp, int nel, typename SubdomainStiffnessMatrix<scalar_t, NB, MX_NEL>::SharedResources &smem) const
        {
            cuddh_assert(blockDim.x == NB * NB * NB && blockDim.y == MX_NEL && blockDim.z == 1,
                         printf("SubdomainStiffnessMatrix<NB = %d, MX_NEL = %d> expects a thread block of dimensions "
                                "(NB^3, MX_NEL).\n",
                                NB, MX_NEL));
            return SubdomainStiffnessMatrix<scalar_t, NB, MX_NEL>(smem, subsp, nel, D, G, I);
        }
    };

    template <typename scalar_t>
    class DDStiffnessMatrix3D
    {
    public:
        DDStiffnessMatrix3D(const H1Space3D &fem, const EnsembleSpace3D &efem);

        DeviceDDStiffnessMatrix3D<scalar_t> to_device() const
        {
            return DeviceDDStiffnessMatrix3D<scalar_t>{
                .D = reshape(thrust::raw_pointer_cast(d.data()), n_basis, n_basis),
                .G = reshape(thrust::raw_pointer_cast(g.data()), n_basis, n_basis, n_basis, mx_elem, n_domains),
                .I = d_I};
        }

    private:
        int n_basis;
        int mx_elem;
        int n_domains;
        thrust::device_vector<scalar_t> d;
        thrust::device_vector<SmallSymmetricMatrix<scalar_t, 3>> g;
        TensorWrapper<5, const int> d_I;
    };

    extern template class DDStiffnessMatrix3D<float>;
    extern template class DDStiffnessMatrix3D<double>;
} // namespace cuddh

#endif
