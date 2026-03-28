#pragma once

#include <type_traits>

#include "EnsembleSpace.hpp"
#include "HostDeviceArray.hpp"
#include "SmallMatrix.hpp"
#include "cuddh_config.hpp"
#include "cuddh_error.hpp"
#include "forall.hpp"

namespace cuddh
{
    template <typename scalar_t, int NB, int NEL>
    struct SubdomainStiffnessMatrix
    {
        using vec_t = cuddh::scalar2<scalar_t>;
        using mat_t = SmallSymmetricMatrix<scalar_t, 2>;

        struct SharedResources
        {
            scalar_t D[NB][NB];
            scalar_t u[NB * NB * NEL];
            vec_t grad[NEL][NB][NB];
        };

        SharedResources &smem;
        mat_t geom;
        int I[2][NB];
        int tx, ty;

        __device__ SubdomainStiffnessMatrix(SharedResources &mem, int subsp, int nel,
                                            const MatrixWrapper<const scalar_t> &D,
                                            const TensorWrapper<4, const mat_t> &G,
                                            const TensorWrapper<4, const int> &sI)
            : smem{mem}
        {
            cuddh_assert(blockDim.x == NB * NB && blockDim.y == NEL && blockDim.z == 1,
                         printf("SubsdomainStiffnessMatrix<NB=%d,NEL=%d> expects a thread block of dimensions (NB^2, "
                                "NEL) = (%d, %d) but got (%d, %d, %d).\n",
                                NB, NEL, NB * NB, NEL, blockDim.x, blockDim.y, blockDim.z));

            tx = threadIdx.x % NB;
            ty = threadIdx.x / NB;
            const auto el = threadIdx.y;

            if (el == 0)
                smem.D[tx][ty] = D(tx, ty);

            for (int i = 0; i < NB; ++i)
            {
                I[0][i] = (el < nel) ? sI(i, ty, el, subsp) : -1;
                I[1][i] = (el < nel) ? sI(tx, i, el, subsp) : -1;
            }

            geom = (el < nel) ? G(tx, ty, el, subsp) : mat_t{};

            __syncthreads();
        }

        __device__ scalar_t operator()(scalar_t in) const
        {
            const auto el = threadIdx.y;

            smem.u[threadIdx.x + NB * NB * threadIdx.y] = in;
            __syncthreads();

            vec_t grad{0, 0};

            if (I[0][0] >= 0) // el < nel
            {
                for (int i = 0; i < NB; ++i)
                {
                    grad.x += smem.D[tx][i] * smem.u[I[0][i]];
                    grad.y += smem.D[ty][i] * smem.u[I[1][i]];
                }
            }
            __syncthreads();

            smem.grad[el][ty][tx] = geom * grad;
            smem.u[threadIdx.x + NB * NB * threadIdx.y] = 0;
            __syncthreads();

            scalar_t Su = 0;
            for (int i = 0; i < NB; ++i)
            {
                Su += smem.D[i][tx] * smem.grad[el][ty][i].x + smem.D[i][ty] * smem.grad[el][i][tx].y;
            }
            atomicAdd(smem.u + I[0][tx], Su);
            __syncthreads();

            return smem.u[threadIdx.x + NB * NB * threadIdx.y];
        }
    };

    template <typename scalar_t>
    struct DeviceDDStiffnessMatrix
    {
        using sym2x2 = SmallSymmetricMatrix<scalar_t, 2>;

        MatrixWrapper<const scalar_t> D;
        TensorWrapper<4, const sym2x2> G;
        TensorWrapper<4, const int> I;

        template <int NB, int MX_NEL>
        __device__ SubdomainStiffnessMatrix<scalar_t, NB, MX_NEL> subspace_op(
            int subsp, int nel, typename SubdomainStiffnessMatrix<scalar_t, NB, MX_NEL>::SharedResources &smem) const
        {
            return SubdomainStiffnessMatrix<scalar_t, NB, MX_NEL>(smem, subsp, nel, D, G, I);
        }
    };

    template <typename scalar_t>
    class DDStiffnessMatrix
    {
        static_assert(std::is_same_v<scalar_t, float> || std::is_same_v<scalar_t, double>,
                      "scalar_t must be float or double");

    public:
        using sym2x2 = SmallSymmetricMatrix<scalar_t, 2>;

        DDStiffnessMatrix(const H1Space2D &fem, const EnsembleSpace &efem);

        DeviceDDStiffnessMatrix<scalar_t> to_device() const
        {
            return {.D = reshape(d.device_read(), n_basis, n_basis),
                    .G = reshape(g.device_read(), n_basis, n_basis, mx_elem, n_domains),
                    .I = d_I};
        }

    private:
        int n_basis;
        int mx_elem;
        int n_domains;
        HostDeviceArray<scalar_t> d;
        HostDeviceArray<sym2x2> g;
        TensorWrapper<4, const int> d_I;
    };

    extern template class DDStiffnessMatrix<float>;
    extern template class DDStiffnessMatrix<double>;
} // namespace cuddh
