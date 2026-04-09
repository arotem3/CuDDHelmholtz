#include "DDHKernelImpl.hpp"
#include "ThreadBlockMinres.hpp"

using namespace cuddh;

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
    double *const __restrict__ y, const scalar_t *const __restrict__ d_lambda, scalar_t *const __restrict__ d_update)
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

    cuddh_assert(
        blockDim.x == EDOF && blockDim.y == NEL && blockDim.z == 1,
        printf(
            "DDH2D error: Attempting to launch ddh_mr_action_kernel<%d, %d, %d> with invalid blockDim %d x %d x %d.\n",
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

template <typename scalar_t>
struct MRKernelDispatcher
{
    int n_basis, tdof, block_size;
    MRKernelDispatcher(int n_basis, int tdof, int block_size) : n_basis(n_basis), tdof(tdof), block_size(block_size) {}

    template <int NB, int TDOF, int BLOCK_SIZE>
    static void dispatch_kernel(const EnsembleSpace &efem, int g_ndof, int n_lambda, const LambdaDOFData<scalar_t> *B,
                                const DDStiffnessMatrix<scalar_t> &S, const scalar_t *punity,
                                MatrixWrapper<const scalar_t> scaled_mass,
                                MatrixWrapper<const scalar_t> scaled_face_mass, scalar_t omega, const double *fem_in,
                                double *fem_out, const scalar_t *lambda_in, scalar_t *lambda_out, scalar_t *d_work)
    {
        constexpr int NEL = BLOCK_SIZE / (NB * NB);

        if (fem_out)
            dla::zeros(2 * g_ndof, fem_out);
        if (lambda_out)
            dla::zeros(2 * n_lambda, lambda_out);

        auto data = DDHMinResKernelData<scalar_t, NB, NEL, TDOF>::make(n_lambda, g_ndof, efem, B, punity, S,
                                                                       scaled_mass, scaled_face_mass, omega, d_work);
        const int n_domains = efem.size();
        dim3 bs(NB * NB, NEL);
        ddh_mr_action_kernel<scalar_t, NB, NEL, TDOF><<<n_domains, bs>>>(data, fem_in, fem_out, lambda_in, lambda_out);
        CUDDH_CHECK_KERNEL();
    }

    template <int NB, int TDOF, typename... Args>
    void dispatch_blocksize(Args &&...args) const
    {
        switch (block_size)
        {
            case 256:
                dispatch_kernel<NB, TDOF, 256>(std::forward<Args>(args)...);
                break;
            case 512:
                dispatch_kernel<NB, TDOF, 512>(std::forward<Args>(args)...);
                break;
            case 1024:
                dispatch_kernel<NB, TDOF, 1024>(std::forward<Args>(args)...);
                break;
            default:
                cuddh_verify(false, printf("DDH error: block_size (=%d) not supported.\n", block_size));
        }
    }

    template <int NB, typename... Args>
    void dispatch_tdof(Args &&...args) const
    {
        switch (tdof)
        {
            case 1:
                dispatch_blocksize<NB, 1>(std::forward<Args>(args)...);
                break;
            case 2:
                dispatch_blocksize<NB, 2>(std::forward<Args>(args)...);
                break;
            case 3:
                dispatch_blocksize<NB, 3>(std::forward<Args>(args)...);
                break;
            case 4:
                dispatch_blocksize<NB, 4>(std::forward<Args>(args)...);
                break;
            default:
                cuddh_verify(false, printf("DDH error: only tdof (=%d) <= 4\n", tdof));
        }
    }

    template <typename... Args>
    void invoke(Args &&...args) const
    {
        switch (n_basis)
        {
            case 2:
                dispatch_tdof<2>(std::forward<Args>(args)...);
                break;
            case 3:
                dispatch_tdof<3>(std::forward<Args>(args)...);
                break;
            case 4:
                dispatch_tdof<4>(std::forward<Args>(args)...);
                break;
            case 5:
                dispatch_tdof<5>(std::forward<Args>(args)...);
                break;
            case 6:
                dispatch_tdof<6>(std::forward<Args>(args)...);
                break;
            case 7:
                dispatch_tdof<7>(std::forward<Args>(args)...);
                break;
            case 8:
                dispatch_tdof<8>(std::forward<Args>(args)...);
                break;
            default:
                cuddh_verify(false, printf("DDH error: only n_basis (=%d) <= 8 supported\n", n_basis));
        }
    }
};

namespace cuddh::details
{
    template <typename scalar_t>
    void invoke_mr_kernel(int n_basis, int tdof, int block_size, const EnsembleSpace &efem, int g_ndof, int n_lambda,
                          const LambdaDOFData<scalar_t> *B, const DDStiffnessMatrix<scalar_t> &S,
                          const scalar_t *punity, MatrixWrapper<const scalar_t> scaled_mass,
                          MatrixWrapper<const scalar_t> scaled_face_mass, scalar_t omega, const double *fem_in,
                          double *fem_out, const scalar_t *lambda_in, scalar_t *lambda_out, scalar_t *d_work)
    {
        MRKernelDispatcher<scalar_t>(n_basis, tdof, block_size)
            .invoke(efem, g_ndof, n_lambda, B, S, punity, scaled_mass, scaled_face_mass, omega, fem_in, fem_out,
                    lambda_in, lambda_out, d_work);
    }

    template void invoke_mr_kernel<float>(int, int, int, const EnsembleSpace &, int, int, const LambdaDOFData<float> *,
                                          const DDStiffnessMatrix<float> &, const float *, MatrixWrapper<const float>,
                                          MatrixWrapper<const float>, float, const double *, double *, const float *,
                                          float *, float *);

    template void invoke_mr_kernel<double>(int, int, int, const EnsembleSpace &, int, int,
                                           const LambdaDOFData<double> *, const DDStiffnessMatrix<double> &,
                                           const double *, MatrixWrapper<const double>, MatrixWrapper<const double>,
                                           double, const double *, double *, const double *, double *, double *);
} // namespace cuddh::details
