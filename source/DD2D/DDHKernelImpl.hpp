// Internal header — included by DDHSetup.cpp, DDHWHKernels.cpp, DDHMinResKernels.cpp.
// Not part of the public API.
#pragma once

#include "DD2D/DDFaceMassMatrix.hpp"
#include "DD2D/DDH.hpp"
#include "DD2D/DDMassMatrix.hpp"
#include "DD2D/DDStiffnessMatrix.hpp"
#include "DD2D/EnsembleSpace.hpp"
#include "DDWaveHoltz.hpp"
#include "LambdaDOFData.hpp"
#include "SmallMatrix.hpp"

namespace cuddh
{
    struct alignas(4) SubdomainNDOFs
    {
        int16_t ndof;
        int16_t fdof;
    };

    template <typename scalar_t, int NB, int NEL, int TDOF = 1>
    class DDHBaseKernelData
    {
    public:
        __device__ constexpr int subspace() const { return blockIdx.x; }
        __device__ constexpr int thread_index() const { return threadIdx.x + (NB * NB) * threadIdx.y; }
        __device__ constexpr int thread_dof_index(int t = 0) const { return thread_index() + (NB * NB * NEL) * t; }

        __device__ constexpr SubdomainNDOFs subdomain_limits() const
        {
            int ndof = s_ndof[subspace()];
            int fdof = s_fdof[subspace()];

            cuddh_assert(ndof <= std::numeric_limits<int16_t>::max() && fdof <= std::numeric_limits<int16_t>::max(),
                         printf("DDH2D error: subdomains too large."));

            return {static_cast<int16_t>(ndof), static_cast<int16_t>(fdof)};
        }

        __device__ constexpr int global_ndof() const { return g_ndof; }
        __device__ constexpr int n_lambda() const { return g_lambda; }
        __device__ constexpr int subdomain_elems() const { return s_elems[subspace()]; }

        __device__ constexpr int global_index(int t = 0) const
        {
            return gI[thread_dof_index(t) + mx_ndof * subspace()];
        }

        __device__ constexpr scalar_t partition_of_unity(int t = 0) const
        {
            return punity[thread_dof_index(t) + mx_ndof * subspace()];
        }

        __device__ constexpr LambdaDOFData<scalar_t> lambda_dof(int o, int t = 0) const
        {
            int k = o + 2 * thread_dof_index(t) + (2 * mx_fdof) * subspace();
            return B[k];
        }

        __device__ __forceinline__ auto stiffness_matrix(
            typename SubdomainStiffnessMatrix<scalar_t, NB, NEL, TDOF>::SharedResources &smem) const
        {
            scalar_t *work_ptr = nullptr;
            if constexpr (TDOF > 1)
                work_ptr = d_work + subspace() * (NB * NB * NEL * TDOF);
            return _stiffness_matrix.template subspace_op<NB, NEL, TDOF>(subspace(), subdomain_elems(), smem, work_ptr);
        }

        static __host__ DDHBaseKernelData make(int nlambda, int ndof, const EnsembleSpace &efem,
                                               const LambdaDOFData<scalar_t> *B, const scalar_t *punity,
                                               const DDStiffnessMatrix<scalar_t> &stiffness_matrix,
                                               scalar_t *d_work = nullptr)
        {
            return DDHBaseKernelData{.mx_ndof = efem.max_size(),
                                     .mx_fdof = efem.max_fsize(),
                                     .g_ndof = ndof,
                                     .g_lambda = nlambda,
                                     .s_ndof = efem.sizes(MemorySpace::DEVICE).data(),
                                     .s_fdof = efem.fsizes(MemorySpace::DEVICE).data(),
                                     .s_elems = efem.n_elems(MemorySpace::DEVICE).data(),
                                     .gI = efem.global_indices(MemorySpace::DEVICE).data(),
                                     .punity = punity,
                                     .B = B,
                                     .d_work = d_work,
                                     ._stiffness_matrix = stiffness_matrix.to_device()};
        }

    public: // POD fields — must be public for aggregate initialisation
        int mx_ndof;
        int mx_fdof;
        int g_ndof;
        int g_lambda;

        const int *s_ndof;
        const int *s_fdof;
        const int *s_elems;
        const int *gI;
        const scalar_t *punity;
        const LambdaDOFData<scalar_t> *B;
        scalar_t *d_work;

        DeviceDDStiffnessMatrix<scalar_t> _stiffness_matrix;
    };

    // Returns the complex forcing vec_t for DOF slot t, assembled from
    // the global FEM vector x and the lambda interface coefficients.
    template <typename scalar_t, typename Helper>
    __device__ cuddh::scalar2<scalar_t> ddh_load_f(const Helper &helper, const double *x,
                                                    const scalar_t *d_lambda, int t)
    {
        using vec_t = cuddh::scalar2<scalar_t>;

        vec_t F{};

        if (x)
        {
            const int g_idx = helper.global_index(t);
            const scalar_t weight = helper.partition_of_unity(t);

            F.x = weight * x[g_idx];
            F.y = weight * x[helper.global_ndof() + g_idx];
        }

        const int idx = helper.thread_dof_index(t);
        if (d_lambda && idx < helper.subdomain_limits().fdof)
        {
            for (int o = 0; o < 2; ++o)
            {
                const auto [i, j, T] = helper.lambda_dof(o, t);
                if (i < 0)
                    break;

                scalar_t lambda = d_lambda[i];
                scalar_t re = lambda;
                scalar_t im = lambda;

                lambda = d_lambda[j];
                re += lambda;
                im -= lambda;

                lambda = d_lambda[helper.n_lambda() + i];
                re -= lambda;
                im += lambda;

                lambda = d_lambda[helper.n_lambda() + j];
                re += lambda;
                im += lambda;

                re *= scalar_t(0.5);
                im *= scalar_t(0.5);

                F.x += T * re;
                F.y += T * im;
            }
        }

        return F;
    }

    // Writes the subdomain solution vec_t u for DOF slot t back to the global arrays.
    template <typename scalar_t, typename Helper>
    __device__ void ddh_write_output(const Helper &helper, int t, cuddh::scalar2<scalar_t> u, double *y,
                                     const scalar_t *d_lambda, scalar_t *d_update)
    {
        if (y)
        {
            const scalar_t weight = helper.partition_of_unity(t);
            const int g_idx = helper.global_index(t);

            atomicAdd(y + g_idx, double(weight * u.x));
            atomicAdd(y + helper.global_ndof() + g_idx, double(weight * u.y));
        }

        const int idx = helper.thread_dof_index(t);
        if (d_update && idx < helper.subdomain_limits().fdof)
        {
            for (int o = 0; o < 2; ++o)
            {
                const auto [i, j, T] = helper.lambda_dof(o, t);
                if (i < 0)
                    break;

                scalar_t lambda = 0, mu = 0;

                if (d_lambda)
                {
                    lambda = d_lambda[i];
                    mu = d_lambda[helper.n_lambda() + i];
                }

                d_update[j] = -lambda + T * u.y;
                d_update[helper.n_lambda() + j] = -mu - T * u.x;
            }
        }
    }

    namespace details
    {
        template <typename scalar_t>
        void invoke_wh_kernel(int n_basis, int tdof, int block_size, const EnsembleSpace &efem, int g_ndof,
                              int n_lambda, const LambdaDOFData<scalar_t> *B, const DDStiffnessMatrix<scalar_t> &S,
                              const scalar_t *punity, const DDWaveHoltz<scalar_t> &W, int waveholtz_iterations,
                              const double *fem_in, double *fem_out, const scalar_t *lambda_in, scalar_t *lambda_out,
                              scalar_t *d_work);

        template <typename scalar_t>
        void invoke_mr_kernel(int n_basis, int tdof, int block_size, const EnsembleSpace &efem, int g_ndof,
                              int n_lambda, const LambdaDOFData<scalar_t> *B, const DDStiffnessMatrix<scalar_t> &S,
                              const scalar_t *punity, MatrixWrapper<const scalar_t> scaled_mass,
                              MatrixWrapper<const scalar_t> scaled_face_mass, scalar_t omega, const double *fem_in,
                              double *fem_out, const scalar_t *lambda_in, scalar_t *lambda_out, scalar_t *d_work);
    } // namespace details
} // namespace cuddh
