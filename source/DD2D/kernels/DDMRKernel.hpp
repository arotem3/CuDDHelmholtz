#pragma once

#include "ThreadBlockMinres.hpp"
#include "source/DD2D/DDHKernelImpl.hpp"

namespace cuddh::details
{
    template <typename scalar_t, int NB, int NEL, int TDOF = 1>
    class DDHMinResKernelData : public DDHBaseKernelData<scalar_t, NB, NEL, TDOF>
    {
        using Base = DDHBaseKernelData<scalar_t, NB, NEL, TDOF>;

    public:
        __device__ constexpr scalar_t omega() const { return _omega; }

        __device__ constexpr scalar_t mass(int idx, int ndof) const
        {
            return (idx < ndof) ? d_mass(idx, this->subspace()) : scalar_t(0);
        }

        __device__ constexpr scalar_t face_mass(int idx, int fdof) const
        {
            return (idx < fdof) ? d_face_mass(idx, this->subspace()) : scalar_t(0);
        }

        static __host__ DDHMinResKernelData make(int nlambda, int ndof, const EnsembleSpace &efem,
                                                 const LambdaDOFData<scalar_t> *B, const scalar_t *punity,
                                                 const DDStiffnessMatrix<scalar_t> &stiffness_matrix,
                                                 MatrixWrapper<const scalar_t> scaled_mass,
                                                 MatrixWrapper<const scalar_t> scaled_face_mass, scalar_t omega,
                                                 scalar_t *d_work = nullptr)
        {
            DDHMinResKernelData d{};
            static_cast<Base &>(d) = Base::make(nlambda, ndof, efem, B, punity, stiffness_matrix, d_work);
            d._omega = omega;
            d.d_mass = scaled_mass;
            d.d_face_mass = scaled_face_mass;
            return d;
        }

    public: // POD fields
        scalar_t _omega;
        MatrixWrapper<const scalar_t> d_mass;
        MatrixWrapper<const scalar_t> d_face_mass;
    };

    template <typename scalar_t, int NB, int NEL, int TDOF = 1>
    __global__ __launch_bounds__(NB * NB * NEL, (32 * 48) / (NB * NB * NEL)) void ddh_mr_action_kernel(
        const DDHMinResKernelData<scalar_t, NB, NEL, TDOF> helper, const double *const __restrict__ x,
        double *const __restrict__ y, const scalar_t *const __restrict__ d_lambda,
        scalar_t *const __restrict__ d_update)
    {
        constexpr int EDOF = NB * NB;
        [[maybe_unused]] constexpr int BDOF = EDOF * NEL;

        using vec_t = cuddh::scalar2<scalar_t>;
        using re_arr_t = cuda::std::array<scalar_t, TDOF>;  // per-thread, one real component
        using arr_t = cuda::std::array<scalar_t, 2 * TDOF>; // [u_re..., u_im...]
        using BStiffness = SubdomainStiffnessMatrix<scalar_t, NB, NEL, TDOF>;
        using MinresSolver = ThreadBlockMinres<scalar_t, 2 * TDOF, EDOF, NEL>;

        constexpr int MRMaxIter = 10 * EDOF * NEL * TDOF; // >> ndof, but minres will quickly lose orthogonality.
        constexpr scalar_t MRRTol = std::is_same_v<scalar_t, double> ? 1e-12 : 1e-6;
        constexpr scalar_t MRATol = std::is_same_v<scalar_t, double> ? 1e-14 : 1e-7;

        cuddh_assert(blockDim.x == EDOF && blockDim.y == NEL && blockDim.z == 1,
                     printf("DDH2D error: Attempting to launch ddh_mr_action_kernel<%d, %d, %d> with invalid blockDim "
                            "%d x %d x %d.\n",
                            NB, NEL, TDOF, blockDim.x, blockDim.y, blockDim.z));

        const SubdomainNDOFs limits = helper.subdomain_limits();

        cuddh_assert(limits.ndof <= BDOF * TDOF,
                     printf("DDH2D error: exceeded maximum number of subdomain DOFs per thread block (%d > %d)\n",
                            limits.ndof, BDOF * TDOF););
        cuddh_assert(helper.subdomain_elems() <= NEL * TDOF,
                     printf("DDH2D error: exceeded maximum number of elements per subdomain.\n"));

        arr_t b{};
        for (int t = 0; t < TDOF; ++t)
        {
            if (helper.thread_dof_index(t) >= limits.ndof)
                break;
            const vec_t v = ddh_load_f<scalar_t>(helper, x, d_lambda, t);
            b[t] = v.x;
            b[TDOF + t] = -v.y; // symmetrize block system
        }

        __shared__ typename BStiffness::SharedResources stiffness_smem;
        __shared__ typename MinresSolver::SharedResources mr_smem;

        const auto A = helper.stiffness_matrix(stiffness_smem);

        auto invokeA = [&](const re_arr_t &u_in) -> re_arr_t {
            if constexpr (TDOF == 1)
                return {A(u_in[0])};
            else
                return A(u_in);
        };

        // symmetric block Helmholtz operator L = [A-ω²M, ωH; ωH, ω²M-A]
        auto L = [&](const arr_t &u_in) -> arr_t {
            re_arr_t v;
            for (int t = 0; t < TDOF; ++t)
                v[t] = u_in[t];

            if constexpr (TDOF == 1)
                v = re_arr_t{A(v[0])};
            else
                v = A(v);

            arr_t out;
            for (int t = 0; t < TDOF; ++t)
                out[t] = v[t];

            for (int t = 0; t < TDOF; ++t)
                v[t] = u_in[TDOF + t];

            if constexpr (TDOF == 1)
                v = re_arr_t{A(v[0])};
            else
                v = A(v);

            for (int t = 0; t < TDOF; ++t)
                out[TDOF + t] = -v[t];

            const scalar_t omega = helper.omega();
            const scalar_t om2 = omega * omega;

            for (int t = 0; t < TDOF; ++t)
            {
                const int idx = helper.thread_dof_index(t);
                const scalar_t m = om2 * helper.mass(idx, limits.ndof);
                const scalar_t h = omega * helper.face_mass(idx, limits.fdof);

                out[t] += -m * u_in[t] + h * u_in[TDOF + t];
                out[TDOF + t] += h * u_in[t] + m * u_in[TDOF + t];
            }

            return out;
        };

        arr_t u{};
        MinresSolver(mr_smem).solve(u, L, b, MRMaxIter, MRRTol, MRATol);

        for (int t = 0; t < TDOF; ++t)
        {
            if (helper.thread_dof_index(t) >= limits.ndof)
                break;
            const vec_t ut{u[t], u[TDOF + t]};
            ddh_write_output<scalar_t>(helper, t, ut, y, d_lambda, d_update);
        }
    }
} // namespace cuddh::details
