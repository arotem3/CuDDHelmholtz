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

            __device__ SubdomainStiffnessMatrix3DSmemImpl(SharedResources &mem, scalar_t * /*unused*/, int subsp,
                                                          int nel, const MatrixWrapper<const scalar_t> &D,
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

        // TDOF>1 implementation using a global work array, processing elements in passes
        template <typename scalar_t, int TDOF, int NB, int NEL>
        struct SubdomainStiffnessMatrix3DTDOFImpl
        {
            using vec_t = cuddh::scalar3<scalar_t>;
            using mat_t = SmallSymmetricMatrix<scalar_t, 3>;
            using arr_t = cuda::std::array<scalar_t, TDOF>;

            struct SharedResources
            {
                scalar_t D[NB][NB];
                scalar_t u[NEL][NB][NB][NB]; // single-pass buffer, indexed [el][x][y][z]
                vec_t grad[NEL][NB][NB][NB];
            };

            SharedResources &smem;
            scalar_t *const global_work;
            mat_t geom[TDOF];
            int I[TDOF]; // global_work index for this thread's DOF per TDOF slot (-1 if out of range)

            __device__ static constexpr int3 get_index3d(int i)
            {
                int3 idx;
                idx.z = i / (NB * NB);
                idx.y = (i % (NB * NB)) / NB;
                idx.x = (i % (NB * NB)) % NB;
                return idx;
            }

            __device__ SubdomainStiffnessMatrix3DTDOFImpl(SharedResources &mem, scalar_t *global_work, int subsp,
                                                          int nel, const MatrixWrapper<const scalar_t> &D,
                                                          const TensorWrapper<5, const mat_t> &G,
                                                          const TensorWrapper<5, const int> &sI)
                : smem{mem}, global_work{global_work}
            {
                const auto [x, y, z] = get_index3d(threadIdx.x);
                const auto el = threadIdx.y;

                if (el == 0 && z == 0)
                    smem.D[x][y] = D(x, y);

                for (int t = 0; t < TDOF; ++t)
                {
                    const int el_t = el + NEL * t;
                    I[t] = (el_t < nel) ? sI(x, y, z, el_t, subsp) : -1;
                    geom[t] = (el_t < nel) ? G(x, y, z, el_t, subsp) : mat_t{};
                }

                __syncthreads();
            }

            __device__ arr_t operator()(arr_t values) const
            {
                constexpr int EDOF = NB * NB * NB;
                constexpr int BDOF = EDOF * NEL;
                const auto [x, y, z] = get_index3d(threadIdx.x);
                const auto el = threadIdx.y;
                const int tid = threadIdx.x + EDOF * threadIdx.y;

                for (int t = 0; t < TDOF; ++t)
                    global_work[tid + BDOF * t] = values[t];
                __syncthreads();

                for (int t = 0; t < TDOF; ++t)
                {
                    if (I[t] >= 0)
                        smem.u[el][x][y][z] = global_work[I[t]];
                    __syncthreads();

                    vec_t grad{0, 0, 0};

                    if (I[t] >= 0)
                    {
                        for (int i = 0; i < NB; ++i)
                        {
                            grad.x += smem.D[x][i] * smem.u[el][i][y][z];
                            grad.y += smem.D[y][i] * smem.u[el][x][i][z];
                            grad.z += smem.D[z][i] * smem.u[el][x][y][i];
                        }
                    }
                    __syncthreads();

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
