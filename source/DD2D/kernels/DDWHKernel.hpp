#pragma once

#include "source/DD2D/DDHKernelImpl.hpp"

namespace cuddh::details
{
    template <typename scalar_t, int NB, int NEL, int TDOF = 1>
    class DDHWHKernelData : public DDHBaseKernelData<scalar_t, NB, NEL, TDOF>
    {
        using Base = DDHBaseKernelData<scalar_t, NB, NEL, TDOF>;

    public:
        __device__ constexpr int waveholtz_iterations() const { return wh_iterations; }

        __device__ __forceinline__ auto waveholtz(int ndof) const
        {
            return _waveholtz.template subspace_op<TDOF>(this->subspace(), this->thread_index(), ndof);
        }

        static __host__ DDHWHKernelData make(int nlambda, int ndof, const EnsembleSpace &efem,
                                             const LambdaDOFData<scalar_t> *B, const scalar_t *punity,
                                             const DDStiffnessMatrix<scalar_t> &stiffness_matrix,
                                             const DDWaveHoltz<scalar_t> &waveholtz, int wh_iters,
                                             scalar_t *d_work = nullptr)
        {
            DDHWHKernelData d{};
            static_cast<Base &>(d) = Base::make(nlambda, ndof, efem, B, punity, stiffness_matrix, d_work);
            d.wh_iterations = wh_iters;
            d._waveholtz = waveholtz.to_device();
            return d;
        }

    public: // POD fields
        int wh_iterations;
        DeviceDDWaveHoltz<scalar_t> _waveholtz;
    };

    template <typename scalar_t, int NB, int NEL, int TDOF = 1>
    __global__ __launch_bounds__(NB * NB * NEL, (32 * 48) / (NB * NB * NEL)) void ddh_wh_action_kernel(
        const DDHWHKernelData<scalar_t, NB, NEL, TDOF> helper, const double *const __restrict__ x,
        double *const __restrict__ y, const scalar_t *const __restrict__ d_lambda,
        scalar_t *const __restrict__ d_update)
    {
        constexpr int EDOF = NB * NB;
        [[maybe_unused]] constexpr int BDOF = EDOF * NEL;

        using vec_t = cuddh::scalar2<scalar_t>;
        using arr_t = std::conditional_t<TDOF == 1, vec_t, cuda::std::array<vec_t, TDOF>>;
        using BStiffness = SubdomainStiffnessMatrix<scalar_t, NB, NEL, TDOF>;
        using BlockReduce = cub::BlockReduce<scalar_t, EDOF, cub::BLOCK_REDUCE_WARP_REDUCTIONS, NEL>;

        constexpr int WHMaxIter = 100;
        constexpr scalar_t WHRTol = std::is_same_v<scalar_t, double> ? 1e-12 : 1e-6;
        constexpr scalar_t WHATol = std::is_same_v<scalar_t, double> ? 1e-14 : 1e-7;

        cuddh_assert(blockDim.x == EDOF && blockDim.y == NEL && blockDim.z == 1,
                     printf("DDH2D error: Attempting to launch ddh_wh_action_kernel<%d, %d, %d> with invalid blockDim "
                            "%d x %d x %d.\n",
                            NB, NEL, TDOF, blockDim.x, blockDim.y, blockDim.z));

        const SubdomainNDOFs limits = helper.subdomain_limits();

        cuddh_assert(limits.ndof <= BDOF * TDOF,
                     printf("DDH2D error: exceeded maximum number of subdomain DOFs per thread block (%d > %d)\n",
                            limits.ndof, BDOF * TDOF););
        cuddh_assert(helper.subdomain_elems() <= NEL * TDOF,
                     printf("DDH2D error: exceeded maximum number of elements per subdomain.\n"));

        __shared__ typename BStiffness::SharedResources smem;

        const auto A = helper.stiffness_matrix(smem);
        const auto evolve_project = helper.waveholtz(limits.ndof);

        auto get = [](auto &a, [[maybe_unused]] int t) -> decltype(auto) {
            if constexpr (TDOF == 1)
                return a;
            else
                return a[t];
        };

        arr_t f{};
        for (int t = 0; t < TDOF; ++t)
        {
            if (helper.thread_dof_index(t) >= limits.ndof)
                break;

            get(f, t) = ddh_load_f<scalar_t>(helper, x, d_lambda, t);
        }

        arr_t u{};
        if (helper.waveholtz_iterations() > 0)
        {
            for (int it = 0; it < helper.waveholtz_iterations(); ++it)
                u = evolve_project(A, u, f);
        }
        else
        {
            __shared__ typename BlockReduce::TempStorage reduce_work;
            __shared__ scalar_t reduce_result;

            auto dist = [&](const arr_t &a, const arr_t &b) -> scalar_t {
                scalar_t dr = 0;
                for (int t = 0; t < TDOF; ++t)
                {
                    scalar_t dx = get(a, t).x - get(b, t).x;
                    scalar_t dy = get(a, t).y - get(b, t).y;
                    dr += dx * dx + dy * dy;
                }

                dr = BlockReduce(reduce_work).Sum(dr);
                if (helper.thread_index() == 0)
                    reduce_result = sqrt(dr);
                __syncthreads();

                return reduce_result;
            };

            arr_t u1 = evolve_project(A, u, f);
            scalar_t r = dist(u1, u);
            u = u1;

            const scalar_t tol = max(WHRTol * r, WHATol);
            for (int it = 0; it < WHMaxIter && r > tol; ++it)
            {
                u1 = evolve_project(A, u, f);
                r = dist(u1, u);
                u = u1;
            }
        }

        for (int t = 0; t < TDOF; ++t)
        {
            if (helper.thread_dof_index(t) >= limits.ndof)
                break;
            ddh_write_output<scalar_t>(helper, t, get(u, t), y, d_lambda, d_update);
        }
    }
} // namespace cuddh::details
