#include "DD3D/DDH3D.hpp"

#include <map>
#include <vector>

#include "FixedTensorWrapper.hpp"
#include "ThreadBlockMinres.hpp"

using namespace cuddh;

template <typename scalar_t, SubdomainSolver Solver>
struct MakeSolverData3D;

template <typename scalar_t>
struct MakeSolverData3D<scalar_t, SubdomainSolver::WaveHoltz>
{
    static DDSolverData3D<scalar_t, SubdomainSolver::WaveHoltz> make(double omega, const GridFunc3D<double> &a,
                                                                     const EnsembleSpace3D &efem, int wh_iters)
    {
        cuddh_verify(wh_iters != 0,
                     printf("DDH3D error: waveholtz_iterations must be positive or -1 for residual-based stopping.\n"));
        return {make_DDWaveHoltz_3d<scalar_t>(efem, scalar_t(omega), &a), wh_iters};
    }
};

template <typename scalar_t>
struct MakeSolverData3D<scalar_t, SubdomainSolver::MINRES>
{
    static DDSolverData3D<scalar_t, SubdomainSolver::MINRES> make(double omega, const GridFunc3D<double> &a,
                                                                  const EnsembleSpace3D &efem, int /*wh_iters*/)
    {
        GridFunc3D<double> a2 = a.transform([] __device__(double x) -> double { return x * x; });
        return {DDMassMatrix3D<scalar_t>(efem, a2), DDFaceMassMatrix3D<scalar_t>(efem, a), scalar_t(omega)};
    }
};

struct alignas(4) SubdomainNDOFs3D
{
    int16_t ndof;
    int16_t fdof;
};

// ---------------------------------------------------------------------------
// Base kernel data (shared between WaveHoltz and MINRES 3D kernels)
// ---------------------------------------------------------------------------

template <typename scalar_t, int NB, int NEL, int TDOF = 1>
class DDH3DBaseKernelData
{
public:
    static constexpr int EDOF = NB * NB * NB;
    static constexpr int BDOF = EDOF * NEL;

    __device__ constexpr int subspace() const { return blockIdx.x; }

    __device__ constexpr int thread_index(int t = 0) const { return threadIdx.x + EDOF * threadIdx.y + BDOF * t; }

    __device__ constexpr SubdomainNDOFs3D subdomain_limits() const
    {
        int ndof = s_ndof[subspace()];
        int fdof = s_fdof[subspace()];

        cuddh_assert(ndof <= std::numeric_limits<int16_t>::max() && fdof <= std::numeric_limits<int16_t>::max(),
                     printf("DDH3D error: subdomains too large."));

        return {static_cast<int16_t>(ndof), static_cast<int16_t>(fdof)};
    }

    __device__ constexpr int global_ndof() const { return g_ndof; }
    __device__ constexpr int n_lambda() const { return g_lambda; }
    __device__ constexpr int subdomain_elems() const { return s_elems[subspace()]; }

    __device__ constexpr int global_index(int t = 0) const { return gI[thread_index(t) + mx_ndof * subspace()]; }

    __device__ constexpr scalar_t partition_of_unity(int t = 0) const
    {
        return punity[thread_index(t) + mx_ndof * subspace()];
    }

    __device__ constexpr LambdaDOFData<scalar_t> lambda_dof(int o, int t = 0) const
    {
        int k = o + 3 * (thread_index(t) + mx_fdof * subspace());
        return B[k];
    }

    __device__ __forceinline__ auto stiffness_matrix(
        typename SubdomainStiffnessMatrix3D<scalar_t, NB, NEL, TDOF>::SharedResources &smem) const
    {
        scalar_t *work_ptr = nullptr;
        if constexpr (TDOF > 1)
            work_ptr = d_work + subspace() * (BDOF * TDOF);
        return _stiffness_matrix.template subspace_op<NB, NEL, TDOF>(subspace(), subdomain_elems(), smem, work_ptr);
    }

    static __host__ DDH3DBaseKernelData make(int nlambda, int ndof, const EnsembleSpace3D &efem,
                                             const LambdaDOFData<scalar_t> *B, const scalar_t *punity,
                                             const DDStiffnessMatrix3D<scalar_t> &stiffness_matrix,
                                             scalar_t *d_work = nullptr)
    {
        return DDH3DBaseKernelData{.mx_ndof = efem.max_size(),
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

    DeviceDDStiffnessMatrix3D<scalar_t> _stiffness_matrix;
};

// ---------------------------------------------------------------------------
// WaveHoltz kernel data
// ---------------------------------------------------------------------------

template <typename scalar_t, int NB, int NEL, int TDOF = 1>
class DDH3DKernelData : public DDH3DBaseKernelData<scalar_t, NB, NEL, TDOF>
{
    using Base = DDH3DBaseKernelData<scalar_t, NB, NEL, TDOF>;

public:
    __device__ constexpr int waveholtz_iterations() const { return wh_iterations; }

    __device__ __forceinline__ auto waveholtz(int ndof) const
    {
        return _waveholtz.template subspace_op<TDOF>(this->subspace(), this->thread_index(), ndof);
    }

    static __host__ DDH3DKernelData make(int nlambda, int ndof, const EnsembleSpace3D &efem,
                                         const LambdaDOFData<scalar_t> *B, const scalar_t *punity,
                                         const DDStiffnessMatrix3D<scalar_t> &stiffness_matrix,
                                         const DDWaveHoltz<scalar_t> &waveholtz, int wh_iters,
                                         scalar_t *d_work = nullptr)
    {
        DDH3DKernelData d{};
        static_cast<Base &>(d) = Base::make(nlambda, ndof, efem, B, punity, stiffness_matrix, d_work);
        d.wh_iterations = wh_iters;
        d._waveholtz = waveholtz.to_device();
        return d;
    }

public:
    int wh_iterations;
    DeviceDDWaveHoltz<scalar_t> _waveholtz;
};

// ---------------------------------------------------------------------------
// MINRES kernel data
// ---------------------------------------------------------------------------

template <typename scalar_t, int NB, int NEL, int TDOF = 1>
class DDH3DMinResKernelData : public DDH3DBaseKernelData<scalar_t, NB, NEL, TDOF>
{
    using Base = DDH3DBaseKernelData<scalar_t, NB, NEL, TDOF>;

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

    static __host__ DDH3DMinResKernelData make(int nlambda, int ndof, const EnsembleSpace3D &efem,
                                               const LambdaDOFData<scalar_t> *B, const scalar_t *punity,
                                               const DDStiffnessMatrix3D<scalar_t> &stiffness_matrix,
                                               MatrixWrapper<const scalar_t> scaled_mass,
                                               MatrixWrapper<const scalar_t> scaled_face_mass, scalar_t omega,
                                               scalar_t *d_work = nullptr)
    {
        DDH3DMinResKernelData d{};
        static_cast<Base &>(d) = Base::make(nlambda, ndof, efem, B, punity, stiffness_matrix, d_work);
        d._omega = omega;
        d.d_mass = scaled_mass;
        d.d_face_mass = scaled_face_mass;
        return d;
    }

public:
    scalar_t _omega;
    MatrixWrapper<const scalar_t> d_mass;
    MatrixWrapper<const scalar_t> d_face_mass;
};

// ---------------------------------------------------------------------------
// WaveHoltz kernel
// ---------------------------------------------------------------------------

template <typename scalar_t, int NB, int NEL, int TDOF = 1>
__global__ __launch_bounds__(NB * NB * NB * NEL,
                             (32 * CUDDH_WARPS_PER_SM) /
                                 (NB * NB * NB * NEL *
                                  TDOF)) void ddh_action_kernel_3d(const DDH3DKernelData<scalar_t, NB, NEL, TDOF> helper,
                                                                  const double *const __restrict__ x,
                                                                  double *const __restrict__ y,
                                                                  const scalar_t *const __restrict__ d_lambda,
                                                                  scalar_t *const __restrict__ d_update)
{
    constexpr int EDOF = NB * NB * NB;
    [[maybe_unused]] constexpr int BDOF = EDOF * NEL;

    using vec_t = cuddh::scalar2<scalar_t>;
    using arr_t = std::conditional_t<TDOF == 1, vec_t, cuda::std::array<vec_t, TDOF>>;
    using BStiffness = SubdomainStiffnessMatrix3D<scalar_t, NB, NEL, TDOF>;
    using BlockReduce = cub::BlockReduce<scalar_t, EDOF, cub::BLOCK_REDUCE_WARP_REDUCTIONS, NEL>;

    constexpr int WHMaxIter = 100;
    constexpr scalar_t WHRTol = std::is_same_v<scalar_t, double> ? 1e-12 : 1e-6;
    constexpr scalar_t WHATol = std::is_same_v<scalar_t, double> ? 1e-14 : 1e-7;

    cuddh_assert(blockDim.x == EDOF && blockDim.y == NEL && blockDim.z == 1,
                 printf("DDH3D error: Attempting to launch ddh_action_kernel_3d<%d, %d, %d> with invalid blockDim "
                        "%d x %d x %d.\n",
                        NB, NEL, TDOF, blockDim.x, blockDim.y, blockDim.z));

    const SubdomainNDOFs3D limits = helper.subdomain_limits();

    cuddh_assert(limits.ndof <= BDOF * TDOF,
                 printf("DDH3D error: exceeded maximum number of subdomain DOFs per thread block (%d > %d)\n",
                        limits.ndof, BDOF * TDOF););
    cuddh_assert(helper.subdomain_elems() <= NEL * TDOF,
                 printf("DDH3D error: exceeded maximum number of elements per subdomain.\n"));

    auto get = [](auto &a, [[maybe_unused]] int t) -> decltype(auto) {
        if constexpr (TDOF == 1)
            return a;
        else
            return a[t];
    };

    const arr_t f = [&]() -> arr_t {
        arr_t F{};

        for (int t = 0; t < TDOF; ++t)
        {
            const int idx = helper.thread_index(t);
            if (idx >= limits.ndof)
                break;

            if (x)
            {
                const int g_idx = helper.global_index(t);
                const scalar_t weight = helper.partition_of_unity(t);

                get(F, t).x += weight * x[g_idx];
                get(F, t).y += weight * x[helper.global_ndof() + g_idx];
            }

            if (d_lambda && idx < limits.fdof)
            {
                for (int o = 0; o < 3; ++o)
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

                    get(F, t).x += T * re;
                    get(F, t).y += T * im;
                }
            }
        }

        return F;
    }();

    __shared__ typename BStiffness::SharedResources smem;
    const auto A = helper.stiffness_matrix(smem);
    const auto evolve_project = helper.waveholtz(limits.ndof);

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
        const int idx = helper.thread_index(t);
        if (idx >= limits.ndof)
            break;

        if (y)
        {
            const scalar_t weight = helper.partition_of_unity(t);
            const int g_idx = helper.global_index(t);

            atomicAdd(y + g_idx, double(weight * get(u, t).x));
            atomicAdd(y + helper.global_ndof() + g_idx, double(weight * get(u, t).y));
        }

        if (d_update && idx < limits.fdof)
        {
            for (int o = 0; o < 3; ++o)
            {
                const auto [i, j, T] = helper.lambda_dof(o, t);

                scalar_t lambda = 0, mu = 0;

                if (d_lambda && i >= 0)
                {
                    lambda = d_lambda[i];
                    mu = d_lambda[helper.n_lambda() + i];
                }

                if (j >= 0)
                {
                    d_update[j] = -lambda + T * get(u, t).y;
                    d_update[helper.n_lambda() + j] = -mu - T * get(u, t).x;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// MINRES kernel
// ---------------------------------------------------------------------------

template <typename scalar_t, int NB, int NEL, int TDOF = 1>
__global__ __launch_bounds__(NB * NB * NB * NEL, 1024 / (NB * NB * NB * NEL)) void ddh_mr_action_kernel_3d(
    const DDH3DMinResKernelData<scalar_t, NB, NEL, TDOF> helper, const double *const __restrict__ x,
    double *const __restrict__ y, const scalar_t *const __restrict__ d_lambda, scalar_t *const __restrict__ d_update)
{
    constexpr int EDOF = NB * NB * NB;
    [[maybe_unused]] constexpr int BDOF = EDOF * NEL;

    using re_arr_t = cuda::std::array<scalar_t, TDOF>;
    using arr_t = cuda::std::array<scalar_t, 2 * TDOF>;
    using BStiffness = SubdomainStiffnessMatrix3D<scalar_t, NB, NEL, TDOF>;
    using MinresSolver = ThreadBlockMinres<scalar_t, 2 * TDOF, EDOF, NEL>;

    constexpr int MRMaxIter = 10 * EDOF * NEL * TDOF;
    constexpr scalar_t MRRTol = std::is_same_v<scalar_t, double> ? 1e-12 : 1e-6;
    constexpr scalar_t MRATol = std::is_same_v<scalar_t, double> ? 1e-14 : 1e-7;

    cuddh_assert(blockDim.x == EDOF && blockDim.y == NEL && blockDim.z == 1,
                 printf("DDH3D error: Attempting to launch ddh_mr_action_kernel_3d<%d, %d, %d> with invalid blockDim "
                        "%d x %d x %d.\n",
                        NB, NEL, TDOF, blockDim.x, blockDim.y, blockDim.z));

    const SubdomainNDOFs3D limits = helper.subdomain_limits();

    cuddh_assert(limits.ndof <= BDOF * TDOF,
                 printf("DDH3D error: exceeded maximum number of subdomain DOFs per thread block (%d > %d)\n",
                        limits.ndof, BDOF * TDOF););
    cuddh_assert(helper.subdomain_elems() <= NEL * TDOF,
                 printf("DDH3D error: exceeded maximum number of elements per subdomain.\n"));

    // Load forcing b = [f_re, -f_im] (symmetrized block system)
    arr_t b{};
    for (int t = 0; t < TDOF; ++t)
    {
        const int idx = helper.thread_index(t);
        if (idx >= limits.ndof)
            break;

        scalar_t f_re = 0, f_im = 0;

        if (x)
        {
            const int g_idx = helper.global_index(t);
            const scalar_t weight = helper.partition_of_unity(t);

            f_re += weight * x[g_idx];
            f_im += weight * x[helper.global_ndof() + g_idx];
        }

        if (d_lambda && idx < limits.fdof)
        {
            for (int o = 0; o < 3; ++o)
            {
                const auto [i, j, T] = helper.lambda_dof(o, t);
                if (i < 0)
                    break;

                scalar_t lam = d_lambda[i];
                scalar_t re = lam;
                scalar_t im = lam;

                lam = d_lambda[j];
                re += lam;
                im -= lam;

                lam = d_lambda[helper.n_lambda() + i];
                re -= lam;
                im += lam;

                lam = d_lambda[helper.n_lambda() + j];
                re += lam;
                im += lam;

                re *= scalar_t(0.5);
                im *= scalar_t(0.5);

                f_re += T * re;
                f_im += T * im;
            }
        }

        b[t] = f_re;
        b[TDOF + t] = -f_im; // symmetrize block system
    }

    __shared__ typename BStiffness::SharedResources stiffness_smem;
    __shared__ typename MinresSolver::SharedResources mr_smem;

    const auto A = helper.stiffness_matrix(stiffness_smem);

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
            const int idx = helper.thread_index(t);
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
        const int idx = helper.thread_index(t);
        if (idx >= limits.ndof)
            break;

        if (y)
        {
            const scalar_t weight = helper.partition_of_unity(t);
            const int g_idx = helper.global_index(t);

            atomicAdd(y + g_idx, double(weight * u[t]));
            atomicAdd(y + helper.global_ndof() + g_idx, double(weight * u[TDOF + t]));
        }

        if (d_update && idx < limits.fdof)
        {
            for (int o = 0; o < 3; ++o)
            {
                const auto [i, j, T] = helper.lambda_dof(o, t);

                scalar_t lambda = 0, mu = 0;

                if (d_lambda && i >= 0)
                {
                    lambda = d_lambda[i];
                    mu = d_lambda[helper.n_lambda() + i];
                }

                if (j >= 0)
                {
                    d_update[j] = -lambda + T * u[TDOF + t];
                    d_update[helper.n_lambda() + j] = -mu - T * u[t];
                }
            }
        }
    }
}

// Populate _B from the connectivity map.
// B(o, i, p) = {lambda_index, dual_lambda_index, trace_op} for face DOF i of subspace p, slot o (o in [0,3)).
// Returns n_lambda = 2 * n_shared.
template <typename scalar_t>
static int lambda_dofs(thrust::device_vector<LambdaDOFData<scalar_t>> &B, const EnsembleSpace3D &efem, double omega,
                       const GridFunc3D<double> &a)
{
    struct SharedDof
    {
        int subspaces[2];
        int local_dof_indices[2];
        double integral;
    };

    const H1Space3D &fem = efem.h1_space();
    const Mesh3D &mesh = fem.mesh();
    const int n_basis = fem.basis().size();
    const int n_domains = efem.size();
    const int mx_fdof = efem.max_fsize();

    auto shared_faces = efem.shared_faces(MemorySpace::HOST);
    const int n_shared_faces = shared_faces.shape(1);
    auto fI = efem.face_indices(MemorySpace::HOST);
    auto faces = efem.faces(MemorySpace::HOST);
    auto A = a.read(MemorySpace::HOST);

    auto w = fem.basis().quadrature().w(MemorySpace::HOST);
    auto x = fem.basis().quadrature().x(MemorySpace::HOST);

    std::map<int, std::map<int, SharedDof>> shared_dofs;

    for (int s = 0; s < n_shared_faces; ++s)
    {
        const int domain0 = shared_faces(0, s);
        const int domain1 = shared_faces(1, s);
        const int local_face_index0 = shared_faces(2, s);
        const int local_face_index1 = shared_faces(3, s);

        const int global_face0 = faces(local_face_index0, domain0);
        const int global_face1 = faces(local_face_index1, domain1);
        cuddh_verify(global_face0 == global_face1, printf("DDH3D error: shared face indices do not match up.\n"));

        const int pair_key = std::min(domain0, domain1) + n_domains * std::max(domain0, domain1);
        auto &dofs = shared_dofs[pair_key];

        const FaceConnectivity connectivity = mesh.face_connectivity(global_face0);
        const int e0 = connectivity.elements[0];
        const int e1 = connectivity.elements[1];
        const QuadFace face = mesh.face(global_face0);

        for (int j = 0; j < n_basis; ++j)
        {
            for (int i = 0; i < n_basis; ++i)
            {
                const int idx0 = fI(i, j, local_face_index0, domain0);
                const int idx1 = fI(i, j, local_face_index1, domain1);
                const int lkey = (domain0 < domain1) ? idx0 : idx1;

                if (not dofs.contains(lkey))
                {
                    SharedDof dof{};
                    dof.subspaces[0] = domain0;
                    dof.subspaces[1] = domain1;
                    dof.local_dof_indices[0] = idx0;
                    dof.local_dof_indices[1] = idx1;
                    dof.integral = 0.0;
                    dofs[lkey] = dof;
                }

                const auto [x0, y0, z0] = face2vol(n_basis, i, j, connectivity.label[0]);
                const auto [ip, jp] = permute_face_index(n_basis, i, j, connectivity.permutation);
                const auto [x1, y1, z1] = face2vol(n_basis, ip, jp, connectivity.label[1]);

                const double a_left = A(x0, y0, z0, e0);
                const double a_right = A(x1, y1, z1, e1);
                const double wt = w(i) * w(j) * face.measure({x(i), x(j)});

                dofs.at(lkey).integral += wt * (a_left + a_right);
            }
        }
    }

    int n_shared = 0;
    for (const auto &[_, dofs] : shared_dofs)
        n_shared += dofs.size();

    const int n_lambda = 2 * n_shared;

    // leading dim 3: at most 3 boundary faces can share a DOF in 3D
    thrust::host_vector<LambdaDOFData<scalar_t>> h_B(3 * mx_fdof * n_domains, LambdaDOFData<scalar_t>{});

    auto b = reshape(thrust::raw_pointer_cast(h_B.data()), 3, mx_fdof, n_domains);

    int k = 0;
    for (const auto &[_, dofs] : shared_dofs)
    {
        for (const auto &[__, dof] : dofs)
        {
            const scalar_t T = std::sqrt(omega * dof.integral);

            for (const int s : {0, 1})
            {
                const int subspace = dof.subspaces[s];
                const int face_index = dof.local_dof_indices[s];

                for (int o = 0; o < 3; ++o)
                {
                    if (b(o, face_index, subspace).i < 0)
                    {
                        b(o, face_index, subspace) = LambdaDOFData<scalar_t>{
                            .i = (s == 0) ? k : n_shared + k, .j = (s == 0) ? n_shared + k : k, .trOp = T};
                        break;
                    }
                }
            }

            ++k;
        }
    }

    B = h_B;

    return n_lambda;
}

template <typename scalar_t>
static thrust::device_vector<scalar_t> partition_of_unity(const H1Space3D &fem, const EnsembleSpace3D &efem)
{
    MassMatrix3D M(fem);
    DDMassMatrix3D<scalar_t> DDM(efem);

    auto d_m = diagonal_mass(M, MemorySpace::DEVICE);
    auto d_ddm = DDM.to_device();

    const int n_domains = efem.size();
    const int mx_dof = efem.max_size();

    auto sizes = efem.sizes(MemorySpace::DEVICE);
    auto gI = efem.global_indices(MemorySpace::DEVICE);

    thrust::device_vector<scalar_t> P(mx_dof * n_domains, 0);
    auto p = reshape(thrust::raw_pointer_cast(P.data()), mx_dof, n_domains);

    forall(mx_dof * n_domains, [=] __device__(int tid) mutable -> void {
        const int i = tid % mx_dof;
        const int subsp = tid / mx_dof;

        if (i >= sizes(subsp))
            return;

        p(i, subsp) = static_cast<scalar_t>(d_ddm(i, subsp) / d_m[gI(i, subsp)]);
    });

    CUDDH_CUDA_CHECK(cudaDeviceSynchronize()); // ?

    return P;
}

static DDKernelConfig make_valid_config(DDKernelConfig config, int nb, int mx_elems)
{
    int mx_dof = nb * nb * nb * mx_elems;

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
            else if (mx_dof <= 1024)
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
                printf("DDH3D: Kernel configuration with tdof = %d requires %d threads/block which "
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
        int t = (mx_dof + B - 1) / B;

        if (config.tdof <= 0)
            config.tdof = t;
        
        cuddh_verify(config.tdof >= t,
                     printf("DDH3D: Kernel configuration with %d threads/block requires tdof >= %d, but tdof = %d "
                            "was specified. This occured because at least one subdomain has %d elements.\n",
                            B, t, config.tdof, mx_elems));
    }

    cuddh_verify(config.tdof <= 4, printf("DDH: Kernel configuration with tdof > 4 not compiled.\n"));
    return config;
}

template <std::floating_point scalar_t, SubdomainSolver Solver>
DDSubstructuredOperator3D<scalar_t, Solver>::DDSubstructuredOperator3D(const EnsembleSpace3D &efem_, double omega_,
                                                                       const GridFunc3D<double> &a, DDKernelConfig config,
                                                                       int waveholtz_iterations_)
    : Operator<scalar_t>(0),
      DDSolverData3D<scalar_t, Solver>{MakeSolverData3D<scalar_t, Solver>::make(omega_, a, efem_, waveholtz_iterations_)},
      efem{efem_},
      g_ndof{efem_.h1_space().size()},
      g_elem{efem_.h1_space().mesh().n_elem()},
      n_basis{efem_.h1_space().basis().size()},
      S(efem_)
{
    cuddh_verify(n_basis >= 2 && n_basis <= 4, printf("DDH3D error: Only n_basis in [2,3,4] supported.\n"););

    n_domains = efem.size();
    mx_dof = efem.max_size();
    mx_fdof = efem.max_fsize();
    mx_elem_per_dom = efem.max_n_elem();

    kernel_config = make_valid_config(config, n_basis, mx_elem_per_dom);

    if (kernel_config.tdof > 1)
    {
        const int work_size = static_cast<int>(kernel_config.block_size) * kernel_config.tdof * n_domains;
        _work.resize(work_size);
    }

    const H1Space3D &fem = efem.h1_space();
    _partition_of_unity = partition_of_unity<scalar_t>(fem, efem);

    n_lambda = lambda_dofs(_B, efem, omega_, a);
    this->set_size(2 * n_lambda);
}

// ---------------------------------------------------------------------------
// Kernel dispatchers
// ---------------------------------------------------------------------------

template <typename scalar_t>
struct KernelDispatcher3D
{
    int n_basis, tdof, block_size;
    KernelDispatcher3D(int n_basis, int tdof, int block_size) : n_basis(n_basis), tdof(tdof), block_size(block_size) {}

    template <int NB, int TDOF, int BLOCK_SIZE>
    static void dispatch_kernel(const EnsembleSpace3D &efem, const int g_ndof, const int n_lambda,
                                const LambdaDOFData<scalar_t> *B, const DDStiffnessMatrix3D<scalar_t> &stiffness_matrix,
                                const scalar_t *punity, const DDWaveHoltz<scalar_t> &waveholtz, int wh_iterations,
                                const double *const __restrict__ x, double *const __restrict__ y,
                                const scalar_t *const __restrict__ d_lambda, scalar_t *const __restrict__ d_update,
                                scalar_t *d_work)
    {
        constexpr int NEL = BLOCK_SIZE / (NB * NB * NB);

        if (y)
            dla::zeros(2 * g_ndof, y);
        if (d_update)
            dla::zeros(2 * n_lambda, d_update);

        auto data = DDH3DKernelData<scalar_t, NB, NEL, TDOF>::make(n_lambda, g_ndof, efem, B, punity, stiffness_matrix,
                                                                   waveholtz, wh_iterations, d_work);
        const int n_domains = efem.size();
        dim3 block_size(NB * NB * NB, NEL);
        ddh_action_kernel_3d<scalar_t, NB, NEL, TDOF><<<n_domains, block_size>>>(data, x, y, d_lambda, d_update);
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
                cuddh_verify(false,
                             printf("DDH error: block_size (=%d) not supported. Must be one of {256, 512, 1024}.\n",
                                    block_size));
                break;
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
                cuddh_verify(false, printf("DDH3D error: Invalid tdof (=%d). Must be one of {1, 2, 3, 4}\n", tdof));
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
            default:
                cuddh_verify(false, printf("DDH3D error: Invalid n_basis (=%d). Must be one of {2, 3, 4}.\n", n_basis));
        }
    }
};

template <typename scalar_t>
struct KernelDispatcher3DMR
{
    int n_basis, tdof, block_size;
    KernelDispatcher3DMR(int n_basis, int tdof, int block_size) : n_basis(n_basis), tdof(tdof), block_size(block_size) {}

    template <int NB, int TDOF, int BLOCK_SIZE>
    static void dispatch_kernel(const EnsembleSpace3D &efem, const int g_ndof, const int n_lambda,
                                const LambdaDOFData<scalar_t> *B, const DDStiffnessMatrix3D<scalar_t> &stiffness_matrix,
                                const scalar_t *punity, MatrixWrapper<const scalar_t> scaled_mass,
                                MatrixWrapper<const scalar_t> scaled_face_mass, scalar_t omega,
                                const double *const __restrict__ x, double *const __restrict__ y,
                                const scalar_t *const __restrict__ d_lambda, scalar_t *const __restrict__ d_update,
                                scalar_t *d_work)
    {
        constexpr int NEL = BLOCK_SIZE / (NB * NB * NB);

        if (y)
            dla::zeros(2 * g_ndof, y);
        if (d_update)
            dla::zeros(2 * n_lambda, d_update);

        auto data = DDH3DMinResKernelData<scalar_t, NB, NEL, TDOF>::make(n_lambda, g_ndof, efem, B, punity,
                                                                         stiffness_matrix, scaled_mass,
                                                                         scaled_face_mass, omega, d_work);
        const int n_domains = efem.size();
        dim3 block_size(NB * NB * NB, NEL);
        ddh_mr_action_kernel_3d<scalar_t, NB, NEL, TDOF><<<n_domains, block_size>>>(data, x, y, d_lambda, d_update);
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
                cuddh_verify(false,
                             printf("DDH error: block_size (=%d) not supported. Must be one of {256, 512, 1024}.\n",
                                    block_size));
                break;
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
                cuddh_verify(false, printf("DDH3D error: Invalid tdof (=%d). Must be one of {1, 2, 3, 4}\n", tdof));
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
            default:
                cuddh_verify(false, printf("DDH3D error: Invalid n_basis (=%d). Must be one of {2, 3, 4}.\n", n_basis));
        }
    }
};

template <std::floating_point scalar_t, SubdomainSolver Solver>
void DDSubstructuredOperator3D<scalar_t, Solver>::action(const double *fem_in, double *fem_out,
                                                         const scalar_t *lambda_in, scalar_t *lambda_out) const
{
    const LambdaDOFData<scalar_t> *B = thrust::raw_pointer_cast(_B.data());
    const scalar_t *punity = thrust::raw_pointer_cast(_partition_of_unity.data());
    scalar_t *d_work = thrust::raw_pointer_cast(_work.data());

    if constexpr (Solver == SubdomainSolver::WaveHoltz)
    {
        KernelDispatcher3D<scalar_t>(n_basis, kernel_config.tdof, static_cast<int>(kernel_config.block_size))
            .invoke(efem, g_ndof, n_lambda, B, S, punity, this->W, this->waveholtz_iterations, fem_in, fem_out,
                    lambda_in, lambda_out, d_work);
    }
    else
    {
        auto sm = this->mass.to_device();
        auto sfm = this->face_mass.to_device();
        KernelDispatcher3DMR<scalar_t>(n_basis, kernel_config.tdof, static_cast<int>(kernel_config.block_size))
            .invoke(efem, g_ndof, n_lambda, B, S, punity, sm, sfm, this->omega, fem_in, fem_out, lambda_in, lambda_out,
                    d_work);
    }
}

template <std::floating_point scalar_t, SubdomainSolver Solver>
void DDSubstructuredOperator3D<scalar_t, Solver>::action(const scalar_t *x, scalar_t *y) const
{
    action((const double *)nullptr, (double *)nullptr, x, y);
    symmetrize_ddh(n_lambda, x, y);
}

template <std::floating_point scalar_t, SubdomainSolver Solver>
void DDSubstructuredOperator3D<scalar_t, Solver>::rhs(const double *f, scalar_t *b) const
{
    action(f, (double *)nullptr, (const scalar_t *)nullptr, b);
    symmetrize_ddh(n_lambda, (const scalar_t *)nullptr, b);
}

template <std::floating_point scalar_t, SubdomainSolver Solver>
void DDSubstructuredOperator3D<scalar_t, Solver>::postprocess(const scalar_t *lambda, const double *f, double *y) const
{
    action(f, y, lambda, (scalar_t *)nullptr);
}

namespace cuddh
{
    template class DDSubstructuredOperator3D<float>;
    template class DDSubstructuredOperator3D<double>;
    template class DDSubstructuredOperator3D<float, SubdomainSolver::MINRES>;
    template class DDSubstructuredOperator3D<double, SubdomainSolver::MINRES>;
    template class DDH3D<float>;
    template class DDH3D<double>;
    template class DDH3D<float, SubdomainSolver::MINRES>;
    template class DDH3D<double, SubdomainSolver::MINRES>;
} // namespace cuddh
