#pragma once

#include <complex>
#include <cuda/std/array>
#include <cuda/warp>
#include <type_traits>

#include "EnsembleSpace.hpp"
#include "FixedTensorWrapper.hpp"
#include "HostDeviceArray.hpp"
#include "SmallMatrix.hpp"
#include "cuddh_config.hpp"
#include "cuddh_error.hpp"
#include "forall.hpp"

namespace cuddh
{
    template <typename scalar_t, bool Complex>
    class BlockSparseMatrix;
    namespace details
    {
        template <typename scalar_t, int NB, int NEL>
            requires(NB == 2 || NB == 4)
        struct SubdomainStiffnessMatrixWarpImpl;

        template <typename scalar_t, int NB, int NEL>
        struct SubdomainStiffnessMatrixSmemImpl;

        template <typename scalar_t, int TDOF, int NB, int NEL>
        struct SubdomainStiffnessMatrixTDOFImpl;

        template <typename scalar_t, int TDOF, int NB, int NEL>
            requires(NB == 2 || NB == 4)
        struct SubdomainStiffnessMatrixTDOFWarpImpl;

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

        DDStiffnessMatrix(const EnsembleSpace &efem);

        DeviceDDStiffnessMatrix<scalar_t> to_device() const
        {
            return {.D = reshape(d.device_read(), n_basis, n_basis),
                    .G = reshape(g.device_read(), n_basis, n_basis, mx_elem, n_domains),
                    .I = d_I};
        }

        /// @brief Accumulate c * S_p into block p of B for each subdomain p.
        /// B must be in COOAssembly state (after `finalize_pattern()`).
        void assemble(scalar_t c, BlockSparseMatrix<scalar_t, false> &B) const;
        void assemble(std::complex<scalar_t> c, BlockSparseMatrix<scalar_t, true> &B) const;

    private:
        const EnsembleSpace &efem;
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

    template <typename scalar_t, int NEL, int TDOF>
    struct SSMImpl<scalar_t, 2, NEL, TDOF>
    {
        using type = SubdomainStiffnessMatrixTDOFWarpImpl<scalar_t, TDOF, 2, NEL>;
    };

    template <typename scalar_t, int NEL, int TDOF>
    struct SSMImpl<scalar_t, 4, NEL, TDOF>
    {
        using type = SubdomainStiffnessMatrixTDOFWarpImpl<scalar_t, TDOF, 4, NEL>;
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
        static_assert(NB * NB * NEL < std::numeric_limits<int16_t>::max());
        static_assert(NB * NB < std::numeric_limits<int8_t>::max());

        using vec_t = cuddh::scalar2<scalar_t>;
        using mat_t = SmallSymmetricMatrix<scalar_t, 2>;

        struct SharedResources
        {
            scalar_t u[2][NEL * NB * NB];
        };

        SharedResources &smem;
        scalar_t Dxy;
        mat_t geom;
        int16_t I;
        int8_t tx, ty;

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
            const int16_t E = (lane / (NB * NB)) * (NB * NB);
            const int16_t tid = threadIdx.x + NB * NB * threadIdx.y;

            smem.u[0][tid] = in;
            smem.u[1][tid] = 0;
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

            return smem.u[1][tid];
        }
    };

    template <typename scalar_t, int NB, int NEL>
    struct SubdomainStiffnessMatrixSmemImpl
    {
        static_assert(NB * NB * NEL < std::numeric_limits<int16_t>::max());
        static_assert(NB * NB < std::numeric_limits<int8_t>::max());

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
        int16_t I;
        int8_t tx, ty;

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

            I = (el < nel) ? sI(tx, ty, el, subsp) : -1;
            geom = (el < nel) ? G(tx, ty, el, subsp) : mat_t{};

            __syncthreads();
        }

        __device__ scalar_t operator()(scalar_t in) const
        {
            const int16_t el = threadIdx.y;
            const int16_t tid = threadIdx.x + NB * NB * threadIdx.y;

            // shuffle
            smem.u[tid] = in;
            __syncthreads();

            in = (I >= 0) ? smem.u[I] : 0;
            __syncthreads();

            FixedTensorWrapper<scalar_t, NB, NB, NEL> u(smem.u);
            u(tx, ty, el) = in;
            __syncthreads();

            vec_t grad{0, 0};

            for (int i = 0; i < NB; ++i)
            {
                grad.x += smem.D[tx][i] * u(i, ty, el);
                grad.y += smem.D[ty][i] * u(tx, i, el);
            }
            __syncthreads();

            smem.grad[el][ty][tx] = geom * grad;
            smem.u[tid] = 0;
            __syncthreads();

            scalar_t Su = 0;
            for (int i = 0; i < NB; ++i)
            {
                Su += smem.D[i][tx] * smem.grad[el][ty][i].x + smem.D[i][ty] * smem.grad[el][i][tx].y;
            }
            if (I >= 0)
                atomicAdd(smem.u + I, Su);
            __syncthreads();

            return smem.u[tid];
        }
    };

    template <typename scalar_t, int TDOF, int NB, int NEL>
    struct SubdomainStiffnessMatrixTDOFImpl
    {
        static_assert(NB * NB * NEL < std::numeric_limits<int16_t>::max());
        static_assert(NB * NB < std::numeric_limits<int8_t>::max());

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
        int16_t I[TDOF];
        int8_t tx, ty;

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
            const int16_t el = threadIdx.y;
            const int16_t tid = threadIdx.x + EDOF * threadIdx.y;

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

    template <typename scalar_t, int TDOF, int NB, int NEL>
        requires(NB == 2 || NB == 4)
    struct SubdomainStiffnessMatrixTDOFWarpImpl
    {
        static_assert(NB * NB * NEL < std::numeric_limits<int16_t>::max());
        static_assert(NB * NB < std::numeric_limits<int8_t>::max());

        using vec_t = cuddh::scalar2<scalar_t>;
        using mat_t = SmallSymmetricMatrix<scalar_t, 2>;
        using arr_t = cuda::std::array<scalar_t, TDOF>;

        struct SharedResources
        {};

        SharedResources &smem;
        scalar_t *const global_work;
        scalar_t Dxy;
        mat_t geom[TDOF];
        int16_t I[TDOF];
        int8_t tx, ty;

        __device__ SubdomainStiffnessMatrixTDOFWarpImpl(SharedResources &mem, scalar_t *global_work, int subsp, int nel,
                                                        const auto &D, const auto &G, const auto &sI)
            : smem{mem}, global_work{global_work}
        {
            cuddh_assert(blockDim.x == NB * NB && blockDim.y == NEL && blockDim.z == 1,
                         printf("SubsdomainStiffnessMatrix<NB=%d,NEL=%d> expects a thread block of dimensions (NB^2, "
                                "NEL) = (%d, %d) but got (%d, %d, %d).\n",
                                NB, NEL, NB * NB, NEL, blockDim.x, blockDim.y, blockDim.z));

            tx = threadIdx.x % NB;
            ty = threadIdx.x / NB;

            Dxy = D(tx, ty);

            for (int t = 0; t < TDOF; ++t)
            {
                const int el = threadIdx.y + NEL * t;

                I[t] = (el < nel) ? sI(tx, ty, el, subsp) : -1;
                geom[t] = (el < nel) ? G(tx, ty, el, subsp) : mat_t{};
            }
        }

        __device__ arr_t operator()(arr_t values) const
        {
            constexpr int EDOF = NB * NB;
            constexpr int BDOF = EDOF * NEL;

            const auto lane = cuda::ptx::get_sreg_laneid();
            const int16_t E = (lane / EDOF) * EDOF;
            const int16_t tid = threadIdx.x + EDOF * threadIdx.y;

            for (int t = 0; t < TDOF; ++t)
                global_work[tid + BDOF * t] = values[t];
            __syncthreads();

            for (int t = 0; t < TDOF; ++t)
                values[t] = (I[t] >= 0) ? global_work[I[t]] : 0;
            __syncthreads();

            for (int t = 0; t < TDOF; ++t)
            {
                vec_t grad{0, 0};

                for (int i = 0; i < NB; ++i)
                {
                    scalar_t uiy = cuda::device::warp_shuffle_idx(values[t], E + (ty * NB + i));
                    scalar_t Dxi = cuda::device::warp_shuffle_idx(Dxy, E + (i * NB + tx));
                    grad.x += Dxi * uiy;

                    scalar_t uxi = cuda::device::warp_shuffle_idx(values[t], E + (i * NB + tx));
                    scalar_t Dyi = cuda::device::warp_shuffle_idx(Dxy, E + (i * NB + ty));
                    grad.y += Dyi * uxi;
                }

                grad = geom[t] * grad;

                values[t] = 0;
                for (int i = 0; i < NB; ++i)
                {
                    scalar_t gx_iy = cuda::device::warp_shuffle_idx(grad.x, E + (ty * NB + i));
                    scalar_t Dix = cuda::device::warp_shuffle_idx(Dxy, E + (tx * NB + i));
                    values[t] += Dix * gx_iy;

                    scalar_t gy_xi = cuda::device::warp_shuffle_idx(grad.y, E + (i * NB + tx));
                    scalar_t Diy = cuda::device::warp_shuffle_idx(Dxy, E + (ty * NB + i));
                    values[t] += Diy * gy_xi;
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
