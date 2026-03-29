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

template <int NB, int NEL, typename scalar_t>
static void ddh_action(const EnsembleSpace *efem, const int g_ndof, /* global finite element degrees of freedom */
                       const int n_lambda,                          /* number of substructured DOFs (lambda) */
                       const TensorWrapper<3, const int2> B, /* global lambda indices associated with boundary DOF */
                       const TensorWrapper<3, const scalar_t> T,                 /* lambda trace operator */
                       const DeviceDDStiffnessMatrix<scalar_t> stiffness_matrix, /* stiffness_matvec matrix on device */
                       const MatrixWrapper<const scalar_t> punity,               /* partition of unity */
                       const DeviceDDWaveHoltz<scalar_t> waveholtz,              /* WaveHoltz data */
                       const double *const __restrict__ x,                       /* input */
                       double *const __restrict__ y,                             /* output */
                       const scalar_t *const __restrict__ d_lambda,              /* substructured problem variables */
                       scalar_t *const __restrict__ d_update                     /* substructured problem variables */
)
{
    constexpr int wh_maxit = 100;
    constexpr scalar_t wh_tol = std::is_same_v<scalar_t, float> ? 1e-6 : 1e-12;

    const int n_domains = efem->size();

    auto gI = efem->global_indices(MemorySpace::DEVICE); // global solution DOF associated with subdomain DOF
    auto s_dof = efem->sizes(MemorySpace::DEVICE);       // number of subdomain degrees of freedom
    auto s_fdof = efem->fsizes(MemorySpace::DEVICE);     // number of face space degrees of freedom
    auto s_elems = efem->n_elems(MemorySpace::DEVICE);   // number of elements in each subdomain

    if (y)
        dla::zeros(2 * g_ndof, y);
    if (d_update)
        dla::zeros(2 * n_lambda, d_update);

    const scalar_t *g_lambda = (d_lambda) ? d_lambda : nullptr;
    const scalar_t *g_mu = (d_lambda) ? (d_lambda + n_lambda) : nullptr;

    scalar_t *lambda_update = (d_update) ? d_update : nullptr;
    scalar_t *mu_update = (d_update) ? (d_update + n_lambda) : nullptr;

    constexpr int EDOF = NB * NB;

    forall_2d(EDOF, NEL, n_domains, [=] __device__(const int subsp) mutable -> void {
        using vec = cuddh::scalar2<scalar_t>;
        using BStiffness = SubdomainStiffnessMatrix<scalar_t, NB, NEL>;
        using BlockReduce = cub::BlockReduce<scalar_t, EDOF, cub::BLOCK_REDUCE_WARP_REDUCTIONS, NEL>;

        const int tid = threadIdx.x + EDOF * threadIdx.y; // linearized thread id

        // get subspace dimensions
        const int fdof = s_fdof(subsp); // dimension of facespace
        const int ndof = s_dof(subsp);  // dimension of subspace

        cuddh_assert(ndof <= EDOF * NEL,
                     printf("DDH2D error: exceeded maximum number of subdomain DOFs per thread block (%d > %d)\n", ndof,
                            EDOF * NEL););
        cuddh_assert(s_elems(subsp) <= NEL,
                     printf("DDH2D error: exceeded maximum number of elements per subdomain.\n"));

        __shared__ typename BStiffness::SharedResources smem;
        __shared__ typename BlockReduce::TempStorage reduce_work;
        __shared__ scalar_t reduce_result;

        auto dist = [&](vec a, vec b) -> scalar_t {
            scalar_t dx = a.x - b.x;
            scalar_t dy = a.y - b.y;
            scalar_t dr = dx * dx + dy * dy;
            dr = BlockReduce(reduce_work).Sum(dr);
            if (tid == 0)
                reduce_result = sqrt(dr);
            __syncthreads();
            return reduce_result;
        };

        const auto A = stiffness_matrix.template subspace_op<NB, NEL>(subsp, s_elems(subsp), smem);
        const auto evolve_project = waveholtz.subspace_op(subsp, tid, ndof);

        const vec F = [&]() -> vec {
            vec F{0, 0};
            if (x && tid < ndof)
            {
                const int g_idx = gI(tid, subsp);
                const scalar_t weight = punity(tid, subsp);

                F.x += weight * scalar_t(x[g_idx]);
                F.y += weight * scalar_t(x[g_ndof + g_idx]);
            }

            if (d_lambda && tid < fdof)
            {
                for (int o = 0; o < 2; ++o)
                {
                    const auto [i, j] = B(o, tid, subsp);
                    if (i < 0)
                        break;

                    const scalar_t lambda1 = g_lambda[i], lambda2 = g_lambda[j], mu1 = g_mu[i], mu2 = g_mu[j];
                    const scalar_t t = T(o, tid, subsp);

                    F.x += scalar_t(0.5) * t * (lambda1 + lambda2 - mu1 + mu2);
                    F.y += scalar_t(0.5) * t * (lambda1 - lambda2 + mu1 + mu2);
                }
            }

            return F;
        }();

        // WaveHoltz iteration
        vec u = {0, 0};
        vec u1 = evolve_project(A, u, F); // (u,v) are the approx solution of the Helmholtz eq.
        scalar_t r = dist(u1, u);
        u = u1;

        scalar_t tol = max(wh_tol * r, wh_tol);
        for (int it = 0; it < wh_maxit && r > tol; ++it)
        {
            u1 = evolve_project(A, u, F);
            r = dist(u1, u);
            u = u1;
        }

        // update global solution
        if (y && (tid < ndof))
        {
            const scalar_t M = punity(tid, subsp);
            const int g_idx = gI(tid, subsp);

            const double m_u = double(M) * double(u.x);
            atomicAdd(y + g_idx, m_u);

            const double m_v = double(M) * double(u.y);
            atomicAdd(y + g_ndof + g_idx, m_v);
        }

        // update Lambdas
        if (d_update && tid < fdof)
            for (int o = 0; o < 2; ++o)
            {
                const auto [i, j] = B(o, tid, subsp);
                if (i < 0)
                    break;

                scalar_t lambda = scalar_t(0);
                scalar_t mu = scalar_t(0);

                if (d_lambda)
                {
                    lambda = g_lambda[i];
                    mu = g_mu[i];
                }

                const scalar_t t = T(o, tid, subsp);

                lambda_update[j] = -lambda + t * u.y;
                mu_update[j] = -mu - t * u.x;
            }
    });
}

static int lambda_dofs(auto &B, auto &T, const EnsembleSpace &efem, double omega, VectorWrapper<const double> a)
{
    const int n_domains = efem.size();
    const int mx_fdof = efem.max_fsize();

    auto cmap = efem.connectivity_map(MemorySpace::HOST);
    auto gI = efem.global_indices(MemorySpace::HOST);
    const int n_shared = cmap.shape(0);

    B.resize(2 * mx_fdof * n_domains);
    T.resize(2 * mx_fdof * n_domains);
    thrust::fill(B.begin(), B.end(), int2{-1, -1});
    thrust::fill(T.begin(), T.end(), 0.0);

    auto b = reshape(B, 2, mx_fdof, n_domains);
    auto t = reshape(T, 2, mx_fdof, n_domains);

    int n_lambda = 0;
    int k = 0;
    for (const auto &dof : cmap)
    {
        for (const int s : {0, 1})
        {
            const int subspace = dof.subspaces[s];
            const int face_index = dof.local_dof_indices[s];

            for (int o = 0; o < 2; ++o)
            {
                if (b(o, face_index, subspace).x < 0)
                {
                    n_lambda++;
                    b(o, face_index, subspace) = {(s == 0) ? k : n_shared + k, (s == 0) ? n_shared + k : k};
                    t(o, face_index, subspace) = std::sqrt(2.0 * omega * a(gI(face_index, subspace)) * dof.face_mass);
                    break;
                }
            }
        }
        k++;
    }

    cuddh_verify(n_lambda == 2 * n_shared,
                 printf("DDH error: lambda dof computation mismatch (%d != %d)\n", n_lambda, 2 * n_shared));

    return n_lambda;
}

template <typename scalar_t>
static thrust::universal_vector<scalar_t> partition_of_unity(const H1Space2D &fem, const EnsembleSpace &efem)
{
    MassMatrix M(fem);
    DDMassMatrix<double> DDM(fem, efem);

    auto d_m = M.to_device();
    auto d_ddm = DDM.to_device();

    const int n_domains = efem.size();
    const int mx_dof = efem.max_size();

    auto sizes = efem.sizes(MemorySpace::DEVICE);
    auto gI = efem.global_indices(MemorySpace::DEVICE);

    thrust::universal_vector<scalar_t> P(mx_dof * n_domains, 0);
    auto p = reshape(P, mx_dof, n_domains);

    forall_1d(mx_dof, n_domains, [=] __device__(int subsp) mutable -> void {
        const auto &i = threadIdx.x;
        const int ndof = sizes(subsp);
        if (i < ndof)
            p(i, subsp) = static_cast<scalar_t>(d_ddm(i, subsp) / d_m[gI(i, subsp)]);
    });

    CUDDH_CUDA_CHECK(cudaDeviceSynchronize());

    return P;
}

template <typename scalar_t>
DDSubstructedProblem<scalar_t>::DDSubstructedProblem(double omega, const double *h_a, const H1Space2D &fem,
                                                     const EnsembleSpace &efem)
    : g_ndof{fem.size()},
      g_elem{fem.mesh().n_elem()},
      n_basis{fem.basis().size()},
      efem{efem},
      S(fem, efem),
      W{make_DDWaveHoltz_2d<scalar_t>(omega, h_a, fem, efem)}
{
    cuddh_verify(n_basis <= 8, printf("DDH error: Only n_basis(=%d) <= 8 supported.\n", n_basis););

    n_domains = efem.size();

    // determine max subspace dimensions
    mx_dof = efem.max_size();
    cuddh_verify(mx_dof <= CUDDH_DD2D_MX_DOF,
                 printf("DDH error: Subdomain with %d DOFs exceeds maximum DOFs allowed per subdomain (=%d)\n", mx_dof,
                        CUDDH_DD2D_MX_DOF));

    mx_fdof = efem.max_fsize();
    mx_elem_per_dom = efem.max_n_elem();

    n_lambda = lambda_dofs(_B, _T, efem, omega, reshape(h_a, fem.size()));

    _partition_of_unity = partition_of_unity<scalar_t>(fem, efem);
}

template <typename scalar_t>
void DDSubstructedProblem<scalar_t>::action(const double *fem_in, double *fem_out, const scalar_t *lambda_in,
                                            scalar_t *lambda_out) const
{
    cuddh_verify(n_basis <= 8, printf("DDH error: Only n_basis <= 8 supported.\n"));

    auto B = reshape(_B, 2, mx_fdof, n_domains);
    auto T = reshape(_T, 2, mx_fdof, n_domains);
    auto punity = reshape(_partition_of_unity, mx_dof, n_domains);

    // Determine the maximum number of DOFs per subdomain
    int max_dofs = mx_elem_per_dom * n_basis * n_basis;

    // Select the smallest MX_DOF variant that fits
    int mx_dof_variant = 0;
    if (max_dofs <= 256)
    {
        mx_dof_variant = 256;
    }
    else if (max_dofs <= 512)
    {
        mx_dof_variant = 512;
    }
    else if (max_dofs <= 1024)
    {
        mx_dof_variant = 1024;
    }
    else
    {
        cuddh_verify(false, printf("DDH error: Subdomain with %d DOFs exceeds supported maximum (1024)\n", max_dofs));
    }

    using func_t = decltype(&::ddh_action<2, 256 / (2 * 2), scalar_t>);
    func_t actionf = nullptr;
    switch (mx_dof_variant)
    {
        case 256:
            switch (n_basis)
            {
                case 2:
                    actionf = ::ddh_action<2, 256 / (2 * 2), scalar_t>;
                    break;
                case 3:
                    actionf = ::ddh_action<3, 256 / (3 * 3), scalar_t>;
                    break;
                case 4:
                    actionf = ::ddh_action<4, 256 / (4 * 4), scalar_t>;
                    break;
                case 5:
                    actionf = ::ddh_action<5, 256 / (5 * 5), scalar_t>;
                    break;
                case 6:
                    actionf = ::ddh_action<6, 256 / (6 * 6), scalar_t>;
                    break;
                case 7:
                    actionf = ::ddh_action<7, 256 / (7 * 7), scalar_t>;
                    break;
                case 8:
                    actionf = ::ddh_action<8, 256 / (8 * 8), scalar_t>;
                    break;
                default:
                    actionf = nullptr;
                    break;
            }
            break;
        case 512:
            switch (n_basis)
            {
                case 2:
                    actionf = ::ddh_action<2, 512 / (2 * 2), scalar_t>;
                    break;
                case 3:
                    actionf = ::ddh_action<3, 512 / (3 * 3), scalar_t>;
                    break;
                case 4:
                    actionf = ::ddh_action<4, 512 / (4 * 4), scalar_t>;
                    break;
                case 5:
                    actionf = ::ddh_action<5, 512 / (5 * 5), scalar_t>;
                    break;
                case 6:
                    actionf = ::ddh_action<6, 512 / (6 * 6), scalar_t>;
                    break;
                case 7:
                    actionf = ::ddh_action<7, 512 / (7 * 7), scalar_t>;
                    break;
                case 8:
                    actionf = ::ddh_action<8, 512 / (8 * 8), scalar_t>;
                    break;
                default:
                    actionf = nullptr;
                    break;
            }
            break;
        case 1024:
            switch (n_basis)
            {
                case 2:
                    actionf = ::ddh_action<2, 1024 / (2 * 2), scalar_t>;
                    break;
                case 3:
                    actionf = ::ddh_action<3, 1024 / (3 * 3), scalar_t>;
                    break;
                case 4:
                    actionf = ::ddh_action<4, 1024 / (4 * 4), scalar_t>;
                    break;
                case 5:
                    actionf = ::ddh_action<5, 1024 / (5 * 5), scalar_t>;
                    break;
                case 6:
                    actionf = ::ddh_action<6, 1024 / (6 * 6), scalar_t>;
                    break;
                case 7:
                    actionf = ::ddh_action<7, 1024 / (7 * 7), scalar_t>;
                    break;
                case 8:
                    actionf = ::ddh_action<8, 1024 / (8 * 8), scalar_t>;
                    break;
                default:
                    actionf = nullptr;
                    break;
            }
            break;
        default:
            actionf = nullptr;
    }

    cuddh_verify(actionf != nullptr,
                 printf("DDH error: No valid kernel for n_basis=%d, mx_dof_variant=%d\n", n_basis, mx_dof_variant));

    (*actionf)(&efem, g_ndof, n_lambda, B, T, S.to_device(), punity, W.to_device(), fem_in, fem_out, lambda_in,
               lambda_out);
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
