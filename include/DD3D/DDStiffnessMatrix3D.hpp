#pragma once

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/universal_vector.h>

#include <cuda/std/array>
#include <type_traits>

#include "EnsembleSpace3D.hpp"
#include "SmallMatrix.hpp"
#include "cuddh_config.hpp"
#include "cuddh_error.hpp"

namespace cuddh
{
    namespace details
    {
        // TDOF=1 implementation using shared memory
        template <typename scalar_t, int NB, int MX_NEL>
        struct SubdomainStiffnessMatrix3DSmemImpl
        {
            static_assert(NB * NB * NB * MX_NEL < std::numeric_limits<int16_t>::max());
            static_assert(NB < std::numeric_limits<int8_t>::max());

            using vec_t = cuddh::scalar3<scalar_t>;
            using mat_t = SmallSymmetricMatrix<scalar_t, 3>;

            struct SharedResources
            {
                scalar_t D[NB][NB];
                scalar_t remap[MX_NEL * NB * NB * NB];
                scalar_t u[MX_NEL][NB][NB][NB];
                vec_t grad[MX_NEL][NB][NB][NB];
            };

            SharedResources &smem;
            mat_t geom;
            int16_t I;
            int8_t x,y,z;

            __device__ SubdomainStiffnessMatrix3DSmemImpl(SharedResources &mem, scalar_t * /*unused*/, int subsp,
                                                          int nel, const MatrixWrapper<const scalar_t> &D,
                                                          const TensorWrapper<5, const mat_t> &G,
                                                          const TensorWrapper<5, const int> &sI)
                : smem{mem}
            {
                z = threadIdx.x / (NB * NB);
                y = (threadIdx.x % (NB * NB)) / NB;
                x = (threadIdx.x % (NB * NB)) % NB;
                const auto el = threadIdx.y;

                if (el == 0 && z == 0)
                    smem.D[x][y] = D(x, y);

                I = (el < nel) ? sI(x, y, z, el, subsp) : -1;
                geom = (el < nel) ? G(x, y, z, el, subsp) : mat_t{};

                __syncthreads();
            }

            __device__ scalar_t operator()(scalar_t in) const
            {
                const int16_t el = threadIdx.y;
                const int16_t tid = threadIdx.x + blockDim.x * threadIdx.y;

                smem.remap[tid] = in;
                __syncthreads();

                smem.u[el][x][y][z] = (I >= 0) ? smem.remap[I] : 0;
                __syncthreads();

                vec_t grad{0, 0, 0};
                for (int i = 0; i < NB; ++i)
                {
                    grad.x += smem.D[x][i] * smem.u[el][i][y][z];
                    grad.y += smem.D[y][i] * smem.u[el][x][i][z];
                    grad.z += smem.D[z][i] * smem.u[el][x][y][i];
                }

                smem.remap[tid] = 0;
                smem.grad[el][x][y][z] = geom * grad;
                __syncthreads();

                scalar_t Su = 0;
                for (int i = 0; i < NB; ++i)
                {
                    Su += smem.D[i][x] * smem.grad[el][i][y][z].x + smem.D[i][y] * smem.grad[el][x][i][z].y +
                          smem.D[i][z] * smem.grad[el][x][y][i].z;
                }
                atomicAdd(smem.remap + I, Su);
                __syncthreads();

                return smem.remap[tid];
            }
        };

        // TDOF>1 implementation using a global work array, processing elements in passes
        template <typename scalar_t, int TDOF, int NB, int NEL>
        struct SubdomainStiffnessMatrix3DTDOFImpl
        {
            static_assert(NB * NB * NB * NEL < std::numeric_limits<int16_t>::max());
            static_assert(NB < std::numeric_limits<int8_t>::max());

            using vec_t = cuddh::scalar3<scalar_t>;
            using mat_t = SmallSymmetricMatrix<scalar_t, 3>;
            using arr_t = cuda::std::array<scalar_t, TDOF>;

            struct SharedResources
            {
                scalar_t D[NB][NB];
                scalar_t u[NEL][NB][NB][NB];
                vec_t grad[NEL][NB][NB][NB];
            };

            SharedResources &smem;
            scalar_t *const global_work;
            mat_t geom[TDOF];
            int16_t I[TDOF];
            int8_t x, y, z;

            __device__ SubdomainStiffnessMatrix3DTDOFImpl(SharedResources &mem, scalar_t *global_work, int subsp,
                                                          int nel, const MatrixWrapper<const scalar_t> &D,
                                                          const TensorWrapper<5, const mat_t> &G,
                                                          const TensorWrapper<5, const int> &sI)
                : smem{mem}, global_work{global_work}
            {
                z = threadIdx.x / (NB * NB);
                y = (threadIdx.x % (NB * NB)) / NB;
                x = (threadIdx.x % (NB * NB)) % NB;
                const int16_t el = threadIdx.y;

                if (el == 0 && z == 0)
                    smem.D[x][y] = D(x, y);

                for (int t = 0; t < TDOF; ++t)
                {
                    const int16_t el_t = el + NEL * t;
                    I[t] = (el_t < nel) ? sI(x, y, z, el_t, subsp) : -1;
                    geom[t] = (el_t < nel) ? G(x, y, z, el_t, subsp) : mat_t{};
                }

                __syncthreads();
            }

            __device__ arr_t operator()(arr_t values) const
            {
                constexpr int EDOF = NB * NB * NB;
                constexpr int BDOF = EDOF * NEL;

                const int16_t el = threadIdx.y;
                const int16_t tid = threadIdx.x + EDOF * threadIdx.y;

                for (int t = 0; t < TDOF; ++t)
                    global_work[tid + BDOF * t] = values[t];
                __syncthreads();

                for (int t = 0; t < TDOF; ++t)
                {
                    smem.u[el][x][y][z] = (I[t] >= 0) ? global_work[I[t]] : 0;
                    __syncthreads();

                    vec_t grad{0, 0, 0};
                    for (int i = 0; i < NB; ++i)
                    {
                        grad.x += smem.D[x][i] * smem.u[el][i][y][z];
                        grad.y += smem.D[y][i] * smem.u[el][x][i][z];
                        grad.z += smem.D[z][i] * smem.u[el][x][y][i];
                    }

                    smem.grad[el][x][y][z] = geom[t] * grad;
                    __syncthreads();

                    values[t] = 0;
                    for (int i = 0; i < NB; ++i)
                    {
                        values[t] += smem.D[i][x] * smem.grad[el][i][y][z].x + smem.D[i][y] * smem.grad[el][x][i][z].y +
                                     smem.D[i][z] * smem.grad[el][x][y][i].z;
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

        template <typename scalar_t, int NB, int NEL, int TDOF>
        struct SSMImpl3D
        {
            using type = SubdomainStiffnessMatrix3DTDOFImpl<scalar_t, TDOF, NB, NEL>;
        };

        template <typename scalar_t, int NB, int NEL>
        struct SSMImpl3D<scalar_t, NB, NEL, 1>
        {
            using type = SubdomainStiffnessMatrix3DSmemImpl<scalar_t, NB, NEL>;
        };

    } // namespace details

    template <typename scalar_t, int NB, int NEL, int TDOF = 1>
    using SubdomainStiffnessMatrix3D = typename details::SSMImpl3D<scalar_t, NB, NEL, TDOF>::type;

    template <typename scalar_t>
    struct DeviceDDStiffnessMatrix3D
    {
        MatrixWrapper<const scalar_t> D;
        TensorWrapper<5, const SmallSymmetricMatrix<scalar_t, 3>> G;
        TensorWrapper<5, const int> I;

        template <int NB, int MX_NEL, int TDOF = 1, typename SharedResources>
        __forceinline__ __device__ SubdomainStiffnessMatrix3D<scalar_t, NB, MX_NEL, TDOF> subspace_op(
            int subsp, int nel, SharedResources &smem, scalar_t *work = nullptr) const
        {
            cuddh_assert(blockDim.x == NB * NB * NB && blockDim.y == MX_NEL && blockDim.z == 1,
                         printf("SubdomainStiffnessMatrix3D<NB = %d, MX_NEL = %d> expects a thread block of dimensions "
                                "(NB^3, MX_NEL).\n",
                                NB, MX_NEL));
            return SubdomainStiffnessMatrix3D<scalar_t, NB, MX_NEL, TDOF>(smem, work, subsp, nel, D, G, I);
        }
    };

    template <typename scalar_t>
    class DDStiffnessMatrix3D
    {
    public:
        DDStiffnessMatrix3D(const EnsembleSpace3D &efem);

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
