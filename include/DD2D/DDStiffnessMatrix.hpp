#pragma once

#include <cuda/std/array>
#include <cuda/warp>
#include <type_traits>

#include "EnsembleSpace.hpp"
#include "HostDeviceArray.hpp"
#include "SmallMatrix.hpp"
#include "cuddh_config.hpp"
#include "cuddh_error.hpp"
#include "forall.hpp"

namespace cuddh
{
    namespace details
    {
        template <typename scalar_t, int NB, int NEL>
            requires(NB == 2 || NB == 4)
        struct SubdomainStiffnessMatrixWarpImpl;

        template <typename scalar_t, int NB, int NEL>
        struct SubdomainStiffnessMatrixSmemImpl;

        template <typename scalar_t, int TDOF, int NB, int NEL>
        struct SubdomainStiffnessMatrixTDOFImpl;

        template <typename scalar_t, int NB, int NEL, int TDOF>
        struct SSMImpl;

    } // namespace details

    template <typename scalar_t, int NB, int NEL, int TDOF = 1>
    using SubdomainStiffnessMatrix = typename details::SSMImpl<scalar_t, NB, NEL, TDOF>::type;

    template <typename scalar_t>
    struct DeviceDDStiffnessMatrix
    {
        using sym2x2 = SmallSymmetricMatrix<scalar_t, 2>;

        MatrixWrapper<const scalar_t> D;
        TensorWrapper<4, const sym2x2> G;
        TensorWrapper<4, const int> I;

        template <int NB, int MX_NEL, int TDOF = 1, typename SharedResources>
        __forceinline__ __device__ auto subspace_op(int subsp, int nel, SharedResources &smem,
                                                    scalar_t *work = nullptr) const
        {
            return SubdomainStiffnessMatrix<scalar_t, NB, MX_NEL, TDOF>(smem, work, subsp, nel, D, G, I);
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

namespace cuddh::details
{
    template <typename scalar_t, int NEL>
    struct SSMImpl<scalar_t, 2, NEL, 1>
    {
        using type = SubdomainStiffnessMatrixWarpImpl<scalar_t, 2, NEL>;
    };

    template <typename scalar_t, int NEL>
    struct SSMImpl<scalar_t, 4, NEL, 1>
    {
        using type = SubdomainStiffnessMatrixWarpImpl<scalar_t, 4, NEL>;
    };

    template <typename scalar_t, int NB, int NEL>
    struct SSMImpl<scalar_t, NB, NEL, 1>
    {
        using type = SubdomainStiffnessMatrixSmemImpl<scalar_t, NB, NEL>;
    };

    template <typename scalar_t, int NB, int NEL, int TDOF>
    struct SSMImpl
    {
        using type = SubdomainStiffnessMatrixTDOFImpl<scalar_t, TDOF, NB, NEL>;
    };

    template <typename scalar_t, int NB, int NEL>
        requires(NB == 2 || NB == 4)
    struct SubdomainStiffnessMatrixWarpImpl
    {
        using vec_t = cuddh::scalar2<scalar_t>;
        using mat_t = SmallSymmetricMatrix<scalar_t, 2>;

        struct SharedResources
        {
            scalar_t u[2][NEL * NB * NB];
        };

        SharedResources &smem;
        mat_t geom;
        int I;
        int tx, ty;
        scalar_t Dxy;

        __device__ SubdomainStiffnessMatrixWarpImpl(SharedResources &mem, scalar_t *, int subsp, int nel,
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

            Dxy = D(tx, ty);

            I = (el < nel) ? sI(tx, ty, el, subsp) : -1;
            geom = (el < nel) ? G(tx, ty, el, subsp) : mat_t{};
        }

        __device__ scalar_t operator()(scalar_t in) const
        {
            const auto lane = cuda::ptx::get_sreg_laneid();
            const int E = (lane / (NB * NB)) * (NB * NB);

            smem.u[0][threadIdx.x + NB * NB * threadIdx.y] = in;
            smem.u[1][threadIdx.x + NB * NB * threadIdx.y] = 0;
            __syncthreads();

            const scalar_t u = (I >= 0) ? smem.u[0][I] : 0;

            vec_t grad = {0, 0};

            for (int i = 0; i < NB; ++i)
            {
                scalar_t uiy = cuda::device::warp_shuffle_idx(u, E + (ty * NB + i));
                scalar_t Dxi = cuda::device::warp_shuffle_idx(Dxy, E + (i * NB + tx));
                grad.x += Dxi * uiy;

                scalar_t uxi = cuda::device::warp_shuffle_idx(u, E + (i * NB + tx));
                scalar_t Dyi = cuda::device::warp_shuffle_idx(Dxy, E + (i * NB + ty));
                grad.y += Dyi * uxi;
            }

            grad = geom * grad;

            scalar_t Su = 0;
            for (int i = 0; i < NB; ++i)
            {
                scalar_t gx_iy = cuda::device::warp_shuffle_idx(grad.x, E + (ty * NB + i));
                scalar_t Dix = cuda::device::warp_shuffle_idx(Dxy, E + (tx * NB + i));
                Su += Dix * gx_iy;

                scalar_t gy_xi = cuda::device::warp_shuffle_idx(grad.y, E + (i * NB + tx));
                scalar_t Diy = cuda::device::warp_shuffle_idx(Dxy, E + (ty * NB + i));
                Su += Diy * gy_xi;
            }

            if (I >= 0)
                atomicAdd(smem.u[1] + I, Su);
            __syncthreads();

            return smem.u[1][threadIdx.x + NB * NB * threadIdx.y];
        }
    };

    template <typename scalar_t, int NB, int NEL>
    struct SubdomainStiffnessMatrixSmemImpl
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

        __device__ SubdomainStiffnessMatrixSmemImpl(SharedResources &mem, scalar_t *, int subsp, int nel,
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
            if (I[0][0] >= 0)
                atomicAdd(smem.u + I[0][tx], Su);
            __syncthreads();

            return smem.u[threadIdx.x + NB * NB * threadIdx.y];
        }
    };

    template <typename scalar_t, int TDOF, int NB, int NEL>
    struct SubdomainStiffnessMatrixTDOFImpl
    {
        using vec_t = cuddh::scalar2<scalar_t>;
        using mat_t = SmallSymmetricMatrix<scalar_t, 2>;
        using arr_t = cuda::std::array<scalar_t, TDOF>;

        struct SharedResources
        {
            scalar_t D[NB][NB];
            scalar_t u[NEL][NB][NB];
            vec_t grad[NEL][NB][NB];
        };

        SharedResources &smem;
        scalar_t *const global_work;
        mat_t geom[TDOF];
        int I[TDOF];
        int tx, ty;

        __device__ SubdomainStiffnessMatrixTDOFImpl(SharedResources &mem, scalar_t *global_work, int subsp, int nel,
                                                    const auto &D, const auto &G, const auto &sI)
            : smem{mem}, global_work{global_work}
        {
            cuddh_assert(blockDim.x == NB * NB && blockDim.y == NEL && blockDim.z == 1,
                         printf("SubsdomainStiffnessMatrix<NB=%d,NEL=%d> expects a thread block of dimensions (NB^2, "
                                "NEL) = (%d, %d) but got (%d, %d, %d).\n",
                                NB, NEL, NB * NB, NEL, blockDim.x, blockDim.y, blockDim.z));

            tx = threadIdx.x % NB;
            ty = threadIdx.x / NB;

            if (threadIdx.y == 0)
                smem.D[tx][ty] = D(tx, ty);

            for (int t = 0; t < TDOF; ++t)
            {
                const int el = threadIdx.y + NEL * t;

                I[t] = (el < nel) ? sI(tx, ty, el, subsp) : -1;
                geom[t] = (el < nel) ? G(tx, ty, el, subsp) : mat_t{};
            }

            __syncthreads();
        }

        __device__ arr_t operator()(arr_t values) const
        {
            constexpr int EDOF = NB * NB;
            constexpr int BDOF = EDOF * NEL;
            const auto el = threadIdx.y;
            const int tid = threadIdx.x + EDOF * threadIdx.y;

            for (int t = 0; t < TDOF; ++t)
                global_work[tid + BDOF * t] = values[t];
            __syncthreads();

            for (int t = 0; t < TDOF; ++t)
            {
                if (I[t] >= 0)
                    smem.u[el][ty][tx] = global_work[I[t]];
                __syncthreads();

                vec_t grad{0, 0};

                for (int i = 0; i < NB; ++i)
                {
                    grad.x += smem.D[tx][i] * smem.u[el][ty][i];
                    grad.y += smem.D[ty][i] * smem.u[el][i][tx];
                }

                smem.grad[el][ty][tx] = geom[t] * grad;
                __syncthreads();

                values[t] = 0;

                for (int i = 0; i < NB; ++i)
                {
                    values[t] += smem.D[i][tx] * smem.grad[el][ty][i].x;
                    values[t] += smem.D[i][ty] * smem.grad[el][i][tx].y;
                }
            }

            for (int t = 0; t < TDOF; ++t)
                global_work[tid + BDOF * t] = 0;
            __syncthreads();

            for (int t = 0; t < TDOF; ++t)
                if (I[t] >= 0)
                    atomicAdd(global_work + I[t], values[t]);
            __syncthreads();

            for (int t = 0; t < TDOF; ++t)
                values[t] = global_work[tid + BDOF * t];

            return values;
        }
    };
} // namespace cuddh::details
