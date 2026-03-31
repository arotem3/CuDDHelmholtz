#include "DD2D/DDH.hpp"

using namespace cuddh;

// Applies R = 0.5*[1-i, 1+i; 1+i, 1-i] which symmetrizes the DDH operator.
// If x is provided, y <- R * (x - y), otherwise y <- R * y.
template <typename scalar_t>
static void symmetrize(int n, const scalar_t *x, scalar_t *y)
{
    constexpr scalar_t half(0.5);
    const int m = n / 2;

    forall(m, [=] __device__(const int i) mutable -> void {
        const int inds[] = {i, i + m, n + i, n + m + i};

        scalar_t Y[4];
        for (int j = 0; j < 4; ++j)
        {
            Y[j] = y[inds[j]];
            if (x)
                Y[j] = x[inds[j]] - Y[j];
        }

        scalar_t RY[] = {half * (Y[0] + Y[1] + Y[2] - Y[3]), half * (Y[0] + Y[1] - Y[2] + Y[3]),
                         -half * (-Y[0] + Y[1] + Y[2] + Y[3]), -half * (Y[0] - Y[1] + Y[2] + Y[3])};

        for (int j = 0; j < 4; ++j)
            y[inds[j]] = RY[j];
    });
}

struct alignas(4) SubdomainNDOFs
{
    int16_t ndof;
    int16_t fdof;
};

template <typename scalar_t, int NB, int NEL, int TDOF = 1>
class DDHKernelData
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
                     printf("DDH2D error subdomains too large."));

        return {static_cast<int16_t>(ndof), static_cast<int16_t>(fdof)};
    }

    __device__ constexpr int global_ndof() const { return g_ndof; }
    __device__ constexpr int n_lambda() const { return g_lambda; }

    __device__ constexpr int subdomain_elems() const { return s_elems[subspace()]; }
    __device__ constexpr int global_index(int t = 0) const { return gI[thread_dof_index(t) + mx_ndof * subspace()]; }
    __device__ constexpr scalar_t partition_of_unity(int t = 0) const
    {
        return punity[thread_dof_index(t) + mx_ndof * subspace()];
    }

    __device__ constexpr LambdaDOFData<scalar_t> lambda_dof(int o, int t = 0) const
    {
        int k = o + 2 * thread_dof_index(t) + (2 * mx_fdof) * subspace();
        return B[k];
    }

    __device__ __forceinline__ auto waveholtz(int ndof) const
    {
        return _waveholtz.template subspace_op<TDOF>(subspace(), thread_index(), ndof);
    }

    __device__ __forceinline__ auto stiffness_matrix(
        typename SubdomainStiffnessMatrix<scalar_t, NB, NEL, TDOF>::SharedResources &smem) const
    {
        return _stiffness_matrix.template subspace_op<NB, NEL, TDOF>(subspace(), subdomain_elems(), smem,
                                                                     d_work + subspace() * (NB * NB * NEL * TDOF));
    }

    static __host__ DDHKernelData make(int nlambda, int ndof, const EnsembleSpace &efem,
                                       const LambdaDOFData<scalar_t> *B, const scalar_t *punity,
                                       const DDStiffnessMatrix<scalar_t> &stiffness_matrix,
                                       const DDWaveHoltz<scalar_t> &waveholtz, scalar_t *d_work = nullptr)
    {
        return DDHKernelData{.mx_ndof = efem.max_size(),
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
                             ._stiffness_matrix = stiffness_matrix.to_device(),
                             ._waveholtz = waveholtz.to_device()};
    }

public:
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
    DeviceDDWaveHoltz<scalar_t> _waveholtz;
};

template <typename scalar_t, int NB, int NEL, int TDOF>
__global__ __launch_bounds__(NB * NB * NEL) void ddh_action_tdof_kernel(
    const DDHKernelData<scalar_t, NB, NEL, TDOF> helper, const double *const __restrict__ x,
    double *const __restrict__ y, const scalar_t *const __restrict__ d_lambda, scalar_t *const __restrict__ d_update)
{
    constexpr int EDOF = NB * NB;
    constexpr int BDOF = EDOF * NEL;

    using vec_t = cuddh::scalar2<scalar_t>;
    using arr_t = cuda::std::array<vec_t, TDOF>;
    using BStiffness = SubdomainStiffnessMatrix<scalar_t, NB, NEL, TDOF>;
    using BlockReduce = cub::BlockReduce<scalar_t, EDOF, cub::BLOCK_REDUCE_WARP_REDUCTIONS, NEL>;

    constexpr int WHMaxIter = 100;
    constexpr scalar_t WHRTol = std::is_same_v<scalar_t, double> ? 1e-12 : 1e-6;
    constexpr scalar_t WHATol = std::is_same_v<scalar_t, double> ? 1e-14 : 1e-7;

    const SubdomainNDOFs limits = helper.subdomain_limits();

    cuddh_assert(limits.ndof <= BDOF * TDOF,
                 printf("DDH2D error: exceeded maximum number of subdomain DOFs per thread block (%d > %d)\n",
                        limits.ndof, BDOF * TDOF););
    cuddh_assert(helper.subdomain_elems() <= NEL * TDOF,
                 printf("DDH2D error: exceeded maximum number of elements per subdomain.\n"));

    const arr_t f = [&]() -> arr_t {
        arr_t F{};

        for (int t = 0; t < TDOF; ++t)
        {
            const int idx = helper.thread_dof_index(t);
            if (idx >= limits.ndof)
                break;

            if (x)
            {
                const int g_idx = helper.global_index(t);
                const scalar_t weight = helper.partition_of_unity(t);

                F[t].x += weight * x[g_idx];
                F[t].y += weight * x[helper.global_ndof() + g_idx];
            }

            if (d_lambda && idx < limits.fdof)
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

                    F[t].x += T * re;
                    F[t].y += T * im;
                }
            }
        }

        return F;
    }();

    __shared__ typename BStiffness::SharedResources smem;
    __shared__ typename BlockReduce::TempStorage reduce_work;
    __shared__ scalar_t reduce_result;

    auto dist = [&](const arr_t &a, const arr_t &b) -> scalar_t {
        scalar_t dr = 0;
        for (int t = 0; t < TDOF; ++t)
        {
            scalar_t dx = a[t].x - b[t].x;
            scalar_t dy = a[t].y - b[t].y;
            dr += dx * dx + dy * dy;
        }

        dr = BlockReduce(reduce_work).Sum(dr);
        if (helper.thread_index() == 0)
            reduce_result = sqrt(dr);
        __syncthreads();

        return reduce_result;
    };

    const auto A = helper.stiffness_matrix(smem);
    const auto evolve_project = helper.waveholtz(limits.ndof);

    arr_t u{};
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

    for (int t = 0; t < TDOF; ++t)
    {
        const int idx = helper.thread_dof_index(t);
        if (idx >= limits.ndof)
            break;

        if (y)
        {
            const scalar_t weight = helper.partition_of_unity(t);
            const int g_idx = helper.global_index(t);

            atomicAdd(y + g_idx, double(weight * u[t].x));
            atomicAdd(y + helper.global_ndof() + g_idx, double(weight * u[t].y));
        }

        if (d_update && idx < limits.fdof)
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

                d_update[j] = -lambda + T * u[t].y;
                d_update[helper.n_lambda() + j] = -mu - T * u[t].x;
            }
        }
    }
}

template <typename scalar_t, int NB, int NEL, int TDOF>
static void ddh_action_tdof(
    const EnsembleSpace *efem, const int g_ndof,             /* global finite element degrees of freedom */
    const int n_lambda,                                      /* number of substructured DOFs (lambda) */
    const TensorWrapper<3, const LambdaDOFData<scalar_t>> B, /* global lambda indices associated with boundary DOF */
    const DDStiffnessMatrix<scalar_t> &stiffness_matrix,     /* stiffness_matvec matrix on device */
    const MatrixWrapper<const scalar_t> punity,              /* partition of unity */
    const DDWaveHoltz<scalar_t> &waveholtz,                  /* WaveHoltz data */
    const double *const __restrict__ x,                      /* input */
    double *const __restrict__ y,                            /* output */
    const scalar_t *const __restrict__ d_lambda,             /* substructured problem variables */
    scalar_t *const __restrict__ d_update                    /* substructured problem variables */
)
{
    printf("Calling: ddh_action_tdof<%d, %d, %d>\n", NB, NEL, TDOF);

    if (y)
        dla::zeros(2 * g_ndof, y);
    if (d_update)
        dla::zeros(2 * n_lambda, d_update);

    constexpr int BDOF = NB * NB * NEL;
    const int n_domains = efem->size();

    thrust::device_vector<scalar_t> work(BDOF * TDOF * n_domains);
    scalar_t *d_work = thrust::raw_pointer_cast(work.data());

    DDHKernelData<scalar_t, NB, NEL, TDOF> data = DDHKernelData<scalar_t, NB, NEL, TDOF>::make(
        n_lambda, g_ndof, *efem, B, punity, stiffness_matrix, waveholtz, d_work);
    dim3 block_size(NB * NB, NEL);
    ddh_action_tdof_kernel<scalar_t, NB, NEL, TDOF><<<n_domains, block_size>>>(data, x, y, d_lambda, d_update);
    CUDDH_CHECK_KERNEL();
}

template <typename scalar_t, int NB, int NEL>
__global__ __launch_bounds__(NB * NB * NEL) void ddh_action_kernel(const DDHKernelData<scalar_t, NB, NEL> helper,
                                                                   const double *const __restrict__ x,
                                                                   double *const __restrict__ y,
                                                                   const scalar_t *const __restrict__ d_lambda,
                                                                   scalar_t *const __restrict__ d_update)
{
    constexpr int EDOF = NB * NB;

    using vec_t = cuddh::scalar2<scalar_t>;
    using BStiffness = SubdomainStiffnessMatrix<scalar_t, NB, NEL>;
    using BlockReduce = cub::BlockReduce<scalar_t, EDOF, cub::BLOCK_REDUCE_WARP_REDUCTIONS, NEL>;

    constexpr int WHMaxIter = 100;
    constexpr scalar_t WHRTol = std::is_same_v<scalar_t, double> ? 1e-12 : 1e-6;
    constexpr scalar_t WHATol = std::is_same_v<scalar_t, double> ? 1e-14 : 1e-7;

    cuddh_assert(blockDim.x == EDOF && blockDim.y == NEL,
                 printf("DDH2D error: Attempting to launch ddh_action_kernel<%d, %d> with invalid blockDim %d x %d.\n",
                        EDOF, NEL, blockDim.x, blockDim.y));

    const SubdomainNDOFs limits = helper.subdomain_limits();

    cuddh_assert(limits.ndof <= EDOF * NEL,
                 printf("DDH2D error: exceeded maximum number of subdomain DOFs per thread block (%d > %d)\n",
                        limits.ndof, EDOF * NEL););
    cuddh_assert(helper.subdomain_elems() <= NEL,
                 printf("DDH2D error: exceeded maximum number of elements per subdomain (%d > %d).\n",
                        helper.subdomain_elems(), NEL));

    const vec_t f = [&]() -> vec_t {
        vec_t F{0, 0};
        if (x && helper.thread_index() < limits.ndof)
        {
            const int g_idx = helper.global_index();
            const scalar_t weight = helper.partition_of_unity();

            F.x += weight * x[g_idx];
            F.y += weight * x[helper.global_ndof() + g_idx];
        }

        if (d_lambda && helper.thread_index() < limits.fdof)
        {
            for (int o = 0; o < 2; ++o)
            {
                const auto [i, j, T] = helper.lambda_dof(o);
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
    }();

    __shared__ typename BStiffness::SharedResources smem;
    __shared__ typename BlockReduce::TempStorage reduce_work;
    __shared__ scalar_t reduce_result;

    auto dist = [&](vec_t a, vec_t b) -> scalar_t {
        scalar_t dx = a.x - b.x;
        scalar_t dr = dx * dx;
        dx = a.y - b.y;
        dr += dx * dx;

        dr = BlockReduce(reduce_work).Sum(dr);
        if (helper.thread_index() == 0)
            reduce_result = sqrt(dr);
        __syncthreads();

        return reduce_result;
    };

    const auto A = helper.stiffness_matrix(smem);
    const auto evolve_project = helper.waveholtz(limits.ndof);

    vec_t u{0, 0};
    vec_t u1 = evolve_project(A, u, f);
    scalar_t r = dist(u1, u);
    u = u1;

    const scalar_t tol = max(WHRTol * r, WHATol);
    for (int it = 0; it < WHMaxIter && r > tol; ++it)
    {
        u1 = evolve_project(A, u, f);
        r = dist(u1, u);
        u = u1;
    }

    // update global with results
    if (y && helper.thread_index() < limits.ndof)
    {
        const scalar_t weight = helper.partition_of_unity();
        const int g_idx = helper.global_index();

        double value = weight * u.x;
        atomicAdd(y + g_idx, value);

        value = weight * u.y;
        atomicAdd(y + helper.global_ndof() + g_idx, value);
    }

    if (d_update && helper.thread_index() < limits.fdof)
    {
        for (int o = 0; o < 2; ++o)
        {
            const auto [i, j, T] = helper.lambda_dof(o);
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

template <typename scalar_t, int NB, int NEL>
static void ddh_action(
    const EnsembleSpace *efem, const int g_ndof,             /* global finite element degrees of freedom */
    const int n_lambda,                                      /* number of substructured DOFs (lambda) */
    const TensorWrapper<3, const LambdaDOFData<scalar_t>> B, /* global lambda indices associated with boundary DOF */
    const DDStiffnessMatrix<scalar_t> &stiffness_matrix,     /* stiffness_matvec matrix on device */
    const MatrixWrapper<const scalar_t> punity,              /* partition of unity */
    const DDWaveHoltz<scalar_t> &waveholtz,                  /* WaveHoltz data */
    const double *const __restrict__ x,                      /* input */
    double *const __restrict__ y,                            /* output */
    const scalar_t *const __restrict__ d_lambda,             /* substructured problem variables */
    scalar_t *const __restrict__ d_update                    /* substructured problem variables */
)
{
    printf("Calling: ddh_action<%d, %d>\n", NB, NEL);

    if (y)
        dla::zeros(2 * g_ndof, y);
    if (d_update)
        dla::zeros(2 * n_lambda, d_update);

    DDHKernelData<scalar_t, NB, NEL> data =
        DDHKernelData<scalar_t, NB, NEL>::make(n_lambda, g_ndof, *efem, B, punity, stiffness_matrix, waveholtz);
    dim3 block_size(NB * NB, NEL);
    int n_domains = efem->size();
    ddh_action_kernel<scalar_t, NB, NEL><<<n_domains, block_size>>>(data, x, y, d_lambda, d_update);
    CUDDH_CHECK_KERNEL();
}

template <typename scalar_t>
static int lambda_dofs(thrust::device_vector<LambdaDOFData<scalar_t>> &B, const EnsembleSpace &efem, double omega,
                       VectorWrapper<const double> a)
{
    const int n_domains = efem.size();
    const int mx_fdof = efem.max_fsize();

    auto cmap = efem.connectivity_map(MemorySpace::HOST);
    auto gI = efem.global_indices(MemorySpace::HOST);
    const int n_shared = cmap.shape(0);

    thrust::host_vector<LambdaDOFData<scalar_t>> h_B(2 * mx_fdof * n_domains, LambdaDOFData<scalar_t>{});
    auto b = reshape(thrust::raw_pointer_cast(h_B.data()), 2, mx_fdof, n_domains);

    int n_lambda = 2 * n_shared;
    for (int k = 0; k < n_shared; ++k)
    {
        const auto &dof = cmap(k);

        for (const int s : {0, 1})
        {
            const int subspace = dof.subspaces[s];
            const int face_index = dof.local_dof_indices[s];

            for (const int o : {0, 1})
            {
                if (b(o, face_index, subspace).i < 0)
                {
                    const scalar_t T = std::sqrt(2.0 * omega * a(gI(face_index, subspace)) * dof.face_mass);

                    b(o, face_index, subspace) = LambdaDOFData<scalar_t>{
                        .i = (s == 0) ? k : n_shared + k, .j = (s == 0) ? n_shared + k : k, .trOp = T};
                    break;
                }
            }
        }
    }

    B = h_B;

    return n_lambda;
}

static constexpr __device__ int2 get_indices(int t, int2 dims)
{
    return {.x = t % dims.x, .y = t / dims.x};
}

template <typename scalar_t>
static thrust::device_vector<scalar_t> partition_of_unity(const H1Space2D &fem, const EnsembleSpace &efem)
{
    MassMatrix M(fem);
    DDMassMatrix<double> DDM(fem, efem);

    auto d_m = M.to_device();
    auto d_ddm = DDM.to_device();

    const int n_domains = efem.size();
    const int mx_dof = efem.max_size();

    auto sizes = efem.sizes(MemorySpace::DEVICE);
    auto gI = efem.global_indices(MemorySpace::DEVICE);

    thrust::device_vector<scalar_t> P(mx_dof * n_domains, 0);
    auto p = reshape(thrust::raw_pointer_cast(P.data()), mx_dof, n_domains);

    forall(mx_dof * n_domains, [=] __device__(int tid) mutable {
        const auto [i, subsp] = get_indices(tid, {mx_dof, n_domains});

        if (i >= sizes(subsp))
            return;

        p(i, subsp) = d_ddm(i, subsp) / d_m[gI(i, subsp)];
    });

    return P;
}

static DDKernelConfig make_valid_config(DDKernelConfig config, int nb, int mx_elems)
{
    int mx_dof = nb * nb * mx_elems;

    if (config.block_size == DDKernelConfig::Default)
    {
        if (config.tdof <= 0)
        {
            if (mx_dof <= 256)
            {
                config.block_size = DDKernelConfig::t256;
                config.tdof = 1;
            }
            else if (mx_dof <= 512)
            {
                config.block_size = DDKernelConfig::t512;
                config.tdof = 1;
            }
            else if (mx_dof < 1024)
            {
                config.block_size = DDKernelConfig::t1024;
                config.tdof = 1;
            }
            else
            {
                config.block_size = DDKernelConfig::t1024;
                config.tdof = (mx_dof + 1023) / 1024;
            }
        }
        else
        {
            int B = (mx_dof + config.tdof - 1) / config.tdof;
            cuddh_verify(
                B <= 1024,
                printf("DDH: Kernel configuration with tdof = %d requires %d threads/block which "
                       "exceeds the maximum of 1024. This occured because at least one subdomain has %d elements.\n",
                       config.tdof, B, mx_elems));

            if (B <= 256)
                config.block_size = DDKernelConfig::t256;
            else if (B <= 512)
                config.block_size = DDKernelConfig::t512;
            else
                config.block_size = DDKernelConfig::t1024;
        }
    }
    else
    {
        int B = static_cast<int>(config.block_size);
        config.tdof = (mx_dof + B - 1) / B;
    }

    cuddh_verify(config.tdof <= 4, printf("DDH: Kernel configuration with tdof > 4 not compiled.\n"));

    return config;
}

template <typename scalar_t>
DDSubstructedProblem<scalar_t>::DDSubstructedProblem(double omega, const double *h_a, const H1Space2D &fem,
                                                     const EnsembleSpace &efem, DDKernelConfig config)
    : g_ndof{fem.size()},
      g_elem{fem.mesh().n_elem()},
      n_basis{fem.basis().size()},
      efem{efem},
      S(fem, efem),
      W{make_DDWaveHoltz_2d<scalar_t>(omega, h_a, fem, efem)}
{
    n_domains = efem.size();

    mx_fdof = efem.max_fsize();
    mx_elem_per_dom = efem.max_n_elem();
    mx_dof = efem.max_size();

    kernel_config = make_valid_config(config, n_basis, mx_elem_per_dom);

    _partition_of_unity = partition_of_unity<scalar_t>(fem, efem);
    CUDDH_CUDA_CHECK(cudaDeviceSynchronize());
    n_lambda = lambda_dofs(_B, efem, omega, reshape(h_a, fem.size()));
}

// KernelDispatcher struct for clean argument separation and invocation
template <typename scalar_t>
struct KernelDispatcher
{
    int n_basis, tdof, block_size;
    KernelDispatcher(int n_basis, int tdof, int block_size) : n_basis(n_basis), tdof(tdof), block_size(block_size) {}

    template <int NB, int TDOF, int BLOCK_SIZE, typename... Args>
    static void dispatch_kernel(Args &&...args)
    {
        if constexpr (TDOF == 1)
        {
            ::ddh_action<scalar_t, NB, BLOCK_SIZE / (NB * NB)>(std::forward<Args>(args)...);
        }
        else
        {
            ::ddh_action_tdof<scalar_t, NB, BLOCK_SIZE / (NB * NB), TDOF>(std::forward<Args>(args)...);
        }
    }

    template <int NB, int TDOF, typename... Args>
    static void dispatch_blocksize(int block_size, Args &&...args)
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
                cuddh_verify(false, printf("DDH error: Invalid block_size %d\n", block_size));
        }
    }

    template <int NB, typename... Args>
    static void dispatch_tdof(int tdof, int block_size, Args &&...args)
    {
        switch (tdof)
        {
            case 1:
                dispatch_blocksize<NB, 1>(block_size, std::forward<Args>(args)...);
                break;
            case 2:
                dispatch_blocksize<NB, 2>(block_size, std::forward<Args>(args)...);
                break;
            case 3:
                dispatch_blocksize<NB, 3>(block_size, std::forward<Args>(args)...);
                break;
            case 4:
                dispatch_blocksize<NB, 4>(block_size, std::forward<Args>(args)...);
                break;
            default:
                cuddh_verify(false, printf("DDH error: Invalid tdof %d\n", tdof));
        }
    }

    template <typename... Args>
    void invoke(Args &&...args) const
    {
        switch (n_basis)
        {
            case 2:
                dispatch_tdof<2>(tdof, block_size, std::forward<Args>(args)...);
                break;
            case 3:
                dispatch_tdof<3>(tdof, block_size, std::forward<Args>(args)...);
                break;
            case 4:
                dispatch_tdof<4>(tdof, block_size, std::forward<Args>(args)...);
                break;
            case 5:
                dispatch_tdof<5>(tdof, block_size, std::forward<Args>(args)...);
                break;
            case 6:
                dispatch_tdof<6>(tdof, block_size, std::forward<Args>(args)...);
                break;
            case 7:
                dispatch_tdof<7>(tdof, block_size, std::forward<Args>(args)...);
                break;
            case 8:
                dispatch_tdof<8>(tdof, block_size, std::forward<Args>(args)...);
                break;
            default:
                cuddh_verify(false, printf("DDH error: on n_basis (=%d) <= 8 supported\n", n_basis));
        }
    }
};

template <typename scalar_t>
void DDSubstructedProblem<scalar_t>::action(const double *fem_in, double *fem_out, const scalar_t *lambda_in,
                                            scalar_t *lambda_out) const
{
    auto B = reshape(_B, 2, mx_fdof, n_domains);
    auto punity = reshape(_partition_of_unity, mx_dof, n_domains);
    int block_size = static_cast<int>(kernel_config.block_size);
    int tdof = kernel_config.tdof;
    KernelDispatcher<scalar_t>(n_basis, tdof, block_size)
        .invoke(&efem, g_ndof, n_lambda, B, S, punity, W, fem_in, fem_out, lambda_in, lambda_out);
}

template <typename scalar_t>
void DDSubstructedProblem<scalar_t>::action(const scalar_t *x, scalar_t *y) const
{
    action((const double *)nullptr, (double *)nullptr, x, y);
    symmetrize(n_lambda, x, y);
}

template <typename scalar_t>
void DDSubstructedProblem<scalar_t>::rhs(const double *f, scalar_t *b) const
{
    action(f, (double *)nullptr, (const scalar_t *)nullptr, b);
    symmetrize<scalar_t>(n_lambda, nullptr, b);
}

template <typename scalar_t>
void DDSubstructedProblem<scalar_t>::postprocess(const scalar_t *lambda, const double *f, double *y) const
{
    action(f, y, lambda, (scalar_t *)nullptr);
}

namespace cuddh
{
    template class DDSubstructedProblem<float>;
    template class DDSubstructedProblem<double>;
    template class DDH<float>;
    template class DDH<double>;
} // namespace cuddh
