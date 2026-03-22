#include "DD2D/DDH.hpp"

using namespace cuddh;

namespace
{
    template <typename scalar_t>
    struct waveholtz
    {
        int nt;
        scalar_t omega;
        scalar_t dt;
        scalar_t weight;
        scalar_t shift;
        scalar_t theta; // modified time step
        scalar_t sigma; // modified acceleration scaling

        constexpr __host__ __device__ scalar_t filter(scalar_t cs) const { return weight * cs - shift; }
    };

    template <typename scalar_t>
    waveholtz<scalar_t> make_waveholtz(double omega, double dt)
    {
        waveholtz<scalar_t> W;
        W.omega = omega;

        double T = (2 * M_PI) / omega;
        W.nt = std::ceil(T / dt);
        dt = T / W.nt;
        W.dt = dt;

        double tan_omega_dt = std::tan(0.5 * omega * dt);
        double a0 = 0.25 * (1 - tan_omega_dt * tan_omega_dt); // corrected shift

        W.weight = 2.0 / W.nt;
        W.shift = W.weight * a0;
        W.theta = tan_omega_dt / omega;
        W.sigma = std::sin(0.5 * omega * dt) / (0.5 * omega);

        return W;
    }
} // namespace

// computes the complex multiplication (c + i*s) * (x + i*y) and stores the result in x and y.
template <typename scalar_t>
__device__ __forceinline__ static void cxmult(scalar_t &x, scalar_t &y, scalar_t c, scalar_t s)
{
    scalar_t t = x;
    x = c * t - s * y;
    y = s * t + c * y;
}

template <int NB>
static constexpr int2 get_index2d(int i)
{
    return {i % NB, i / NB};
}

template <typename scalar_t, int NB, int NEL>
struct BlockStiffness
{
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
    int I[2][NB];

    __device__ BlockStiffness(SharedResources &mem, int subsp, int nel,
                              const typename DDStiffnessMatrix<scalar_t>::DeviceDDStiffnessMatrix &stiffness_matrix,
                              const TensorWrapper<4, const int> &sI)
        : smem{mem}
    {
        const auto [x, y] = get_index2d<NB>(threadIdx.x);
        const auto el = threadIdx.y;

        if (el == 0)
            smem.D[x][y] = stiffness_matrix.D(x, y);

        for (int i = 0; i < NB; ++i)
        {
            I[0][i] = (el < nel) ? sI(i, y, el, subsp) : -1;
            I[1][i] = (el < nel) ? sI(x, i, el, subsp) : -1;
        }

        geom = (el < nel) ? stiffness_matrix.G(x, y, el, subsp) : mat_t{};

        __syncthreads();
    }

    __device__ scalar_t operator()(scalar_t in) const
    {
        const auto [x, y] = get_index2d<NB>(threadIdx.x);
        const auto el = threadIdx.y;

        smem.u[threadIdx.x + NB * NB * threadIdx.y] = in;
        __syncthreads();

        vec_t grad{0, 0};

        if (I[0][0] >= 0) // el < nel
        {
            for (int i = 0; i < NB; ++i)
            {
                grad.x += smem.D[x][i] * smem.u[I[0][i]];
                grad.y += smem.D[y][i] * smem.u[I[1][i]];
            }
        }
        __syncthreads();

        smem.grad[el][y][x] = geom * grad;
        smem.u[threadIdx.x + NB * NB * threadIdx.y] = 0;
        __syncthreads();

        scalar_t Su = 0;
        for (int i = 0; i < NB; ++i)
        {
            Su += smem.D[i][x] * smem.grad[el][y][i].x + smem.D[i][y] * smem.grad[el][i][x].y;
        }
        atomicAdd(smem.u + I[0][x], Su);
        __syncthreads();

        return smem.u[threadIdx.x + NB * NB * threadIdx.y];
    }
};

template <int NB, int NEL, typename scalar_t>
static void ddh_action(
    const EnsembleSpace *efem, const int g_ndof, /* global finite element degrees of freedom */
    const int n_lambda,                          /* number of substructured DOFs (lambda) */
    const TensorWrapper<3, const int2> B,        /* global lambda indices associated with boundary DOF */
    const TensorWrapper<3, const scalar_t> T,    /* lambda trace operator */
    const typename DDStiffnessMatrix<scalar_t>::DeviceDDStiffnessMatrix
        stiffness_matrix,                                   /* stiffness_matvec matrix on device */
    const MatrixWrapper<const cuddh::scalar2<scalar_t>> ab, /* alpha and beta coefficients for time stepping */
    const MatrixWrapper<const scalar_t> punity,             /* partition of unity */
    const waveholtz<scalar_t> W,                            /* WaveHoltz data */
    const double *const __restrict__ x,                     /* input */
    double *const __restrict__ y,                           /* output */
    const scalar_t *const __restrict__ d_lambda,            /* substructured problem variables */
    scalar_t *const __restrict__ d_update                   /* substructured problem variables */
)
{
    constexpr int wh_maxit = 20;

    const int n_domains = efem->size();

    auto gI = efem->global_indices(MemorySpace::DEVICE);   // global solution DOF associated with subdomain DOF
    auto sI = efem->subspace_indices(MemorySpace::DEVICE); // mapping from (i,j)-node on element to subspace DOF
    auto s_dof = efem->sizes(MemorySpace::DEVICE);         // number of subdomain degrees of freedom
    auto s_fdof = efem->fsizes(MemorySpace::DEVICE);       // number of face space degrees of freedom
    auto s_elems = efem->n_elems(MemorySpace::DEVICE);     // number of elements in each subdomain

    if (y)
        dla::zeros(2 * g_ndof, y);

    const scalar_t *g_lambda = (d_lambda) ? d_lambda : nullptr;
    const scalar_t *g_mu = (d_lambda) ? (d_lambda + n_lambda) : nullptr;

    scalar_t *lambda_update = (d_update) ? d_update : nullptr;
    scalar_t *mu_update = (d_update) ? (d_update + n_lambda) : nullptr;

    // R = Rx + i Ry = exp(0.5 * i * omega * dt) used in the short term recurrence:
    // z(t + 0.5 * dt) = R * z(t) where z(t) = exp(i * omega * t).
    const scalar_t Rx = static_cast<scalar_t>(std::cos(0.5 * W.omega * W.dt));
    const scalar_t Ry = static_cast<scalar_t>(std::sin(0.5 * W.omega * W.dt));

    constexpr int EDOF = NB * NB;

    forall_2d(EDOF, NEL, n_domains, [=] __device__(const int subsp) mutable -> void {
        using BStiffness = BlockStiffness<scalar_t, NB, NEL>;

        const int tid = threadIdx.x + EDOF * threadIdx.y; // linearized thread id

        // get subspace dimensions
        const int fdof = s_fdof(subsp); // dimension of facespace
        const int ndof = s_dof(subsp);  // dimension of subspace

        cuddh_assert(ndof <= CUDDH_DD2D_MX_DOF,
                     printf("DDH2D error: exceeded maximum number of subdomain DOFs per thread block (%d > %d)\n", ndof,
                            CUDDH_DD2D_MX_DOF););
        cuddh_assert(s_elems(subsp) <= NEL,
                     printf("DDH2D error: exceeded maximum number of elements per subdomain.\n"));

        __shared__ typename BStiffness::SharedResources smem;
        BStiffness A(smem, subsp, s_elems(subsp), stiffness_matrix, sI);

        const scalar_t sigma = W.sigma; // modified time step in WaveHoltz iteration
        const auto [alpha, beta] = ab(tid, subsp);

        const cuddh::scalar2<scalar_t> F = [&]() -> cuddh::scalar2<scalar_t> {
            cuddh::scalar2<scalar_t> F{scalar_t(0), scalar_t(0)};
            if (x && tid < ndof)
            {
                const int g_idx = gI(tid, subsp);
                const scalar_t weight = punity(tid, subsp);

                F.x = weight * scalar_t(x[g_idx]);
                F.y = weight * scalar_t(x[g_ndof + g_idx]);
            }

            if (d_lambda && tid < fdof)
            {
                for (int o = 0; o < 2; ++o)
                {
                    int idx = B(o, tid, subsp).x;
                    if (idx >= 0)
                    {
                        F.x += g_lambda[idx];
                        F.y += g_mu[idx];
                    }
                }
            }

            return F;
        }();

        auto evolve_project = [&](cuddh::scalar2<scalar_t> u) -> cuddh::scalar2<scalar_t> {
            scalar_t cs = scalar_t(1);
            scalar_t sn = scalar_t(0);
            scalar_t K = W.filter(cs);

            scalar_t p = u.x;

            cxmult(cs, sn, Rx, Ry);
            scalar_t q = (-sn * u.x + cs * u.y) * W.omega;

            u.x = K * p;

            K = W.filter(cs);
            u.y = K * q;

            // time stepping
            for (int n = 1; n < W.nt; ++n)
            {
                cxmult(cs, sn, Rx, Ry);
                K = W.filter(cs);

                // update p and u
                p += sigma * q;
                u.x += K * p;

                // update q and v
                q = alpha * q + beta * (-A(p) + cs * F.x + sn * F.y);

                cxmult(cs, sn, Rx, Ry);
                K = W.filter(cs);

                u.y += K * q;
            } // time stepping

            // rescale v
            u.y /= W.omega;

            return u;
        };

        // WaveHoltz iteration
        cuddh::scalar2<scalar_t> u{0, 0}; // (u,v) are the approx solution of the Helmholtz eq.
        int it = 0;
        for (; it < wh_maxit; ++it)
        {
            u = evolve_project(u);
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

                scalar_t lambda = scalar_t(0);
                scalar_t mu = scalar_t(0);

                if (d_lambda && i >= 0)
                {
                    lambda = g_lambda[i];
                    mu = g_mu[i];
                }

                if (j >= 0)
                {
                    const scalar_t t = T(o, tid, subsp);
                    lambda_update[j] = -lambda + t * u.y;
                    mu_update[j] = -mu - t * u.x;
                }
            }
    });
}

static int lambda_dofs(auto &B, auto &T, const EnsembleSpace &efem, double omega, auto a)
{
    const int n_domains = efem.size();
    const int mx_fdof = efem.max_fsize();

    auto cmap = efem.connectivity_map(MemorySpace::HOST);
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
            for (int o = 0; o < 2; ++o)
            {
                const int subspace = dof.subspaces[s];
                const int face_index = dof.local_dof_indices[s];

                if (b(o, face_index, subspace).x < 0)
                {
                    n_lambda++;
                    b(o, face_index, subspace) = {(s == 0) ? k : n_shared + k, (s == 0) ? n_shared + k : k};
                    t(o, face_index, subspace) = 2.0 * omega * a(face_index, subspace) * dof.face_mass;
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

// map a global dofs to the subdomain dofs
template <typename T1, typename T2>
static void DD_gridfun(T1 *h_u_dd, const T2 *h_u_mesh, const EnsembleSpace *efem)
{
    const int n_domains = efem->size();
    const int mx_dof = efem->max_size();

    auto sizes = efem->sizes(MemorySpace::HOST);
    auto gI = efem->global_indices(MemorySpace::HOST);

    auto dd = reshape(h_u_dd, mx_dof, n_domains);

    for (int subsp = 0; subsp < n_domains; ++subsp)
    {
        const int ndof = sizes(subsp);
        for (int i = 0; i < ndof; ++i)
            dd(i, subsp) = h_u_mesh[gI(i, subsp)];
    }
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
static thrust::universal_vector<cuddh::scalar2<scalar_t>> make_alpha_beta(double omega, double dt,
                                                                          const MatrixWrapper<scalar_t> a,
                                                                          const H1Space2D &fem,
                                                                          const EnsembleSpace &efem)
{
    DDMassMatrix<scalar_t> M(fem, efem);
    DDFaceMassMatrix<scalar_t> H(fem, efem);

    auto m = M.to_device();
    auto h = H.to_device();

    const int n_domains = efem.size();
    const int mx_dof = efem.max_size();

    auto s_dof = efem.sizes(MemorySpace::DEVICE);   // number of subdomain degrees of freedom
    auto s_fdof = efem.fsizes(MemorySpace::DEVICE); // number of face space degrees of freedom

    constexpr int n = CUDDH_DD2D_MX_DOF; // number of threads per block
    thrust::universal_vector<cuddh::scalar2<scalar_t>> ab(n * n_domains);
    auto alpha_beta = reshape(ab, n, n_domains);

    waveholtz<scalar_t> W = make_waveholtz<scalar_t>(omega, dt);

    forall_1d(n, n_domains, [=] __device__(int subsp) mutable -> void {
        const auto &i = threadIdx.x;
        const int ndof = s_dof(subsp);  // dimension of subspace
        const int fdof = s_fdof(subsp); // dimension of facespace

        cuddh::scalar2<scalar_t> ab{0, 0};

        if (i < ndof)
        {
            scalar_t ai = a(i, subsp);                            // variable coefficient
            scalar_t Mi = m(i, subsp);                            // subdomain mass matrix
            scalar_t Hi = (i < fdof) ? h(i, subsp) : scalar_t(0); // subdomain boundary face mass matrix

            Hi *= ai;
            Mi *= ai * ai;

            const scalar_t inv = 1 / (Mi + W.theta * Hi);
            ab.x = (Mi - W.theta * Hi) * inv;
            ab.y = W.sigma * inv;
        }

        alpha_beta(i, subsp) = ab;
    });

    return ab;
}

template <typename scalar_t>
DDSubstructedProblem<scalar_t>::DDSubstructedProblem(double omega_, const double *h_a, const H1Space2D &fem,
                                                     const EnsembleSpace &efem)
    : g_ndof{fem.size()},
      g_elem{fem.mesh().n_elem()},
      n_basis{fem.basis().size()},
      omega{omega_},
      efem{efem},
      S(fem, efem)
{
    cuddh_verify(n_basis <= 8, printf("DDH error: Only n_basis <= 8 supported.\n"););

    n_domains = efem.size();

    // determine max subspace dimensions
    mx_dof = efem.max_size();
    mx_fdof = efem.max_fsize();
    mx_elem_per_dom = efem.max_n_elem();

    thrust::universal_vector<scalar_t> _a(mx_dof * n_domains);
    auto a = reshape(_a, mx_dof, n_domains);
    DD_gridfun(a.data(), h_a, &efem);

    n_lambda = lambda_dofs(_B, _T, efem, omega, a);

    _partition_of_unity = partition_of_unity<scalar_t>(fem, efem);

    // time step determined by CFL condition: dt = C * h / (n_basis * n_basis * max_vel)
    const double h = fem.mesh().min_h();
    const double reciprocal_max_vel = *std::min_element(h_a, h_a + g_ndof);
    dt = 2.0 * reciprocal_max_vel * h / (n_basis * n_basis);

    alpha_beta = make_alpha_beta<scalar_t>(omega, dt, a, fem, efem);
}

template <typename scalar_t>
void DDSubstructedProblem<scalar_t>::action(const double *fem_in, double *fem_out, const scalar_t *lambda_in,
                                            scalar_t *lambda_out) const
{
    cuddh_verify(n_basis <= 8, printf("DDH error: Only n_basis <= 8 supported.\n"););

    auto B = reshape(_B, 2, mx_fdof, n_domains);
    auto T = reshape(_T, 2, mx_fdof, n_domains);

    auto ab = reshape(alpha_beta, CUDDH_DD2D_MX_DOF, n_domains);

    auto punity = reshape(_partition_of_unity, mx_dof, n_domains);

    using func_t = decltype(&::ddh_action<4, CUDDH_DD2D_MX_DOF / (4 * 4), scalar_t>);
    auto actionf = [&]() -> func_t {
        switch (n_basis)
        {
            case 2:
                return ::ddh_action<2, CUDDH_DD2D_MX_DOF / (2 * 2), scalar_t>;
            case 3:
                return ::ddh_action<3, CUDDH_DD2D_MX_DOF / (3 * 3), scalar_t>;
            case 4:
                return ::ddh_action<4, CUDDH_DD2D_MX_DOF / (4 * 4), scalar_t>;
            case 5:
                return ::ddh_action<5, CUDDH_DD2D_MX_DOF / (5 * 5), scalar_t>;
            case 6:
                return ::ddh_action<6, CUDDH_DD2D_MX_DOF / (6 * 6), scalar_t>;
            case 7:
                return ::ddh_action<7, CUDDH_DD2D_MX_DOF / (7 * 7), scalar_t>;
            case 8:
                return ::ddh_action<8, CUDDH_DD2D_MX_DOF / (8 * 8), scalar_t>;
            default:
                return (func_t) nullptr;
                break;
        }
    }();

    (*actionf)(&efem, g_ndof, n_lambda, B, T, S.to_device(), ab, punity, make_waveholtz<scalar_t>(omega, dt), fem_in,
               fem_out, lambda_in, lambda_out);
}

template <typename scalar_t>
void DDSubstructedProblem<scalar_t>::action(const scalar_t *x, scalar_t *y) const
{
    dla::zeros(2 * n_lambda, y);
    action((const double *)nullptr, (double *)nullptr, x, y);
    dla::axpby(2 * n_lambda, scalar_t(1), x, scalar_t(-1), y);
}

template <typename scalar_t>
void DDSubstructedProblem<scalar_t>::rhs(const double *f, scalar_t *b) const
{
    dla::zeros(2 * n_lambda, b);
    action(f, (double *)nullptr, (const scalar_t *)nullptr, b);
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
