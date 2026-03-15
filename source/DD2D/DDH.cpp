#include "DD2D/DDH.hpp"

// one dimensional size of each domain decomp block. Each block has
// DDH_BLOCK_SIZE * DDH_BLOCK_SIZE degrees of freedom.
#define DDH_BLOCK_SIZE 16

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

template <typename Map, typename Key>
inline static bool contains(const Map &map, Key key)
{
    return map.find(key) != map.end();
}

// computes the complex multiplication (c + i*s) * (x + i*y) and stores the result in x and y.
template <typename scalar_t>
__device__ __forceinline__ static void cxmult(scalar_t &x, scalar_t &y, scalar_t c, scalar_t s)
{
    scalar_t t = x;
    x = c * t - s * y;
    y = s * t + c * y;
}

template <int NB, typename scalar_t>
__device__ __forceinline__ static void stiffness_matvec(scalar_t *const s_u, const SmallMatrix<scalar_t, 2, 2> &geom,
                                                        const int (&Ix)[NB], const int (&Iy)[NB],
                                                        const scalar_t (&s_D)[NB][NB],
                                                        cuddh::scalar2<scalar_t> s_w[][NB][NB])
{
    const auto &[k, l, el] = threadIdx;

    cuddh::scalar2<scalar_t> grad{scalar_t(0), scalar_t(0)};

#pragma unroll NB
    for (int i = 0; i < NB; ++i)
        grad.x += s_D[k][i] * s_u[Ix[i]];

#pragma unroll NB
    for (int i = 0; i < NB; ++i)
        grad.y += s_D[l][i] * s_u[Iy[i]];
    __syncthreads();

    s_w[el][l][k] = geom * grad;

    s_u[k + NB * (l + NB * el)] = scalar_t(0); // zero out output buffer
    __syncthreads();

    scalar_t Su = scalar_t(0);

#pragma unroll NB
    for (int i = 0; i < NB; ++i)
        Su += s_D[i][k] * s_w[el][l][i].x;

#pragma unroll NB
    for (int i = 0; i < NB; ++i)
        Su += s_D[i][l] * s_w[el][i][k].y;

    atomicAdd(s_u + Ix[k], Su);
    __syncthreads();
}

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
    const int n_dupl = B.shape(0);

    auto gI = efem->global_indices(MemorySpace::DEVICE);   // global solution DOF associated with subdomain DOF
    auto sI = efem->subspace_indices(MemorySpace::DEVICE); // mapping from (i,j)-node on element to subspace DOF
    auto s_dof = efem->sizes(MemorySpace::DEVICE);         // number of subdomain degrees of freedom
    auto s_fdof = efem->fsizes(MemorySpace::DEVICE);       // number of face space degrees of freedom

    const scalar_t rw = scalar_t(1) / W.omega;

    if (y)
        dla::zeros(2 * g_ndof, y);

    const scalar_t *g_lambda = (d_lambda) ? d_lambda : nullptr;
    const scalar_t *g_mu = (d_lambda) ? (d_lambda + n_lambda) : nullptr;

    scalar_t *lambda_update = (d_update) ? d_update : nullptr;
    scalar_t *mu_update = (d_update) ? (d_update + n_lambda) : nullptr;

    constexpr int MX_NDOF = NB * NB * NEL * NEL; // == DDH_BLOCK_SIZE^2

    // R = Rx + i Ry = exp(0.5 * i * omega * dt) used in the short term recurrence:
    // z(t + 0.5 * dt) = R * z(t) where z(t) = exp(i * omega * t).
    const scalar_t Rx = static_cast<scalar_t>(std::cos(0.5 * W.omega * W.dt));
    const scalar_t Ry = static_cast<scalar_t>(std::sin(0.5 * W.omega * W.dt));

    forall_3d(NB, NB, NEL * NEL, n_domains, [=] __device__(const int subsp) mutable -> void {
        using BlockReduce = cub::BlockReduce<scalar_t, NB * NB * NEL * NEL>;

        const auto &[k, l, el] = threadIdx;     // convinient indicies
        const int tid = k + NB * (l + NB * el); // linearized thread id

        // get subspace dimensions
        const int fdof = s_fdof(subsp); // dimension of facespace
        const int ndof = s_dof(subsp);  // dimension of subspace

        cuddh_assert(ndof <= MX_NDOF,
                     printf("DDH2D error: exceeded maximum number of subdomain DOFs per thread block (%d > %d)\n", ndof,
                            MX_NDOF););

        // shared mem
        __shared__ scalar_t s_p[MX_NDOF];
        __shared__ cuddh::scalar2<scalar_t> s_work[NEL * NEL][NB][NB];
        __shared__ scalar_t s_D[NB][NB];

        // copy D
        if (tid < NB * NB)
            s_D[k][l] = stiffness_matrix.D(k, l);

        int Ix[NB]; // indices for slice used d/dx computation
        int Iy[NB]; // indices for slice used d/dy computation

#pragma unroll NB
        for (int i = 0; i < NB; ++i)
            Ix[i] = sI(i, l, el, subsp);

#pragma unroll NB
        for (int i = 0; i < NB; ++i)
            Iy[i] = sI(k, i, el, subsp);

        cuddh_assert(Ix[k] == Iy[l], printf("DDH2D error: invalid mapping Ix[k] (%d) != Iy[l] (%d)\n", Ix[k], Iy[l]););

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
                for (int o = 0; o < n_dupl; ++o)
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

        cuddh::scalar2<scalar_t> u{scalar_t(0), scalar_t(0)}; // (u,v) are the approx solution of the Helmholtz eq.

        const auto geom = stiffness_matrix.G(tid, subsp);

        // returns A * x where A is the stiffness matrix
        auto A = [&](scalar_t x) -> scalar_t {
            s_p[tid] = x;
            __syncthreads();
            stiffness_matvec<NB, scalar_t>(s_p, geom, Ix, Iy, s_D, s_work);
            return s_p[tid];
        };

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
            u.y *= rw;

            return u;
        };

        // WaveHoltz iteration
        int it = 0;
        for (; it < wh_maxit; ++it)
        {
            u = evolve_project(u);
        } // WaveHoltz

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
            for (int o = 0; o < n_dupl; ++o)
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

static std::pair<int, int> lambda_dofs(auto &B, auto &T, const EnsembleSpace &efem, double omega, auto a)
{
    const int n_domains = efem.size();
    const int mx_fdof = efem.max_fsize();

    auto cmap = efem.connectivity_map(MemorySpace::HOST);
    const int n_shared = cmap.shape(0);

    int n_dupl = 1;
    std::vector<std::unordered_map<int, int>> counter(n_domains);

    for (const auto &dof : cmap)
    {
        for (int i = 0; i < 2; ++i)
        {
            int subspace = dof.subspaces[i];
            int face_index = dof.local_dof_indices[i];
            int &count = counter[subspace][face_index];
            count++;
            n_dupl = std::max(n_dupl, count);
        }
    }

    B.resize(n_dupl * mx_fdof * n_domains);
    T.resize(n_dupl * mx_fdof * n_domains);
    thrust::fill(B.begin(), B.end(), int2{-1, -1});
    thrust::fill(T.begin(), T.end(), 0.0);

    auto b = reshape(B, n_dupl, mx_fdof, n_domains);
    auto t = reshape(T, n_dupl, mx_fdof, n_domains);

    int n_lambda = 0;
    int k = 0;
    for (const auto &dof : cmap)
    {
        for (const int s : {0, 1})
        {
            for (int o = 0; o < n_dupl; ++o)
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

    return {n_lambda, n_dupl};
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

    constexpr int n = DDH_BLOCK_SIZE * DDH_BLOCK_SIZE; // number of threads per block
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
    cuddh_verify(n_basis == 4 || n_basis == 8, printf("DDH error: Only n_basis==4, and n_basis==8 supported.\n"););

    n_domains = efem.size();

    // determine max subspace dimensions
    mx_dof = efem.max_size();
    mx_fdof = efem.max_fsize();
    mx_elem_per_dom = efem.max_n_elem();

    thrust::universal_vector<scalar_t> _a(mx_dof * n_domains);
    auto a = reshape(_a, mx_dof, n_domains);
    DD_gridfun(a.data(), h_a, &efem);

    std::tie(n_lambda, n_dupl) = lambda_dofs(_B, _T, efem, omega, a);

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
    auto B = reshape(_B, n_dupl, mx_fdof, n_domains);
    auto T = reshape(_T, n_dupl, mx_fdof, n_domains);

    auto d_S = S.to_device();
    auto ab = reshape(alpha_beta, DDH_BLOCK_SIZE * DDH_BLOCK_SIZE, n_domains);

    auto punity = reshape(_partition_of_unity, mx_dof, n_domains);

    auto actionf = [&]() {
        using func_t = decltype(&::ddh_action<4, DDH_BLOCK_SIZE / 4, scalar_t>);

        cuddh_verify(n_basis == 4 || n_basis == 8, printf("DDH error: Only n_basis==4, and n_basis==8 supported.\n"););

        if (n_basis == 4)
            return &::ddh_action<4, DDH_BLOCK_SIZE / 4, scalar_t>;
        else if (n_basis == 8)
            return &::ddh_action<8, DDH_BLOCK_SIZE / 8, scalar_t>;
        return (func_t) nullptr;
    }();

    (*actionf)(&efem, g_ndof, n_lambda, B, T, d_S, ab, punity, make_waveholtz<scalar_t>(omega, dt), fem_in, fem_out,
               lambda_in, lambda_out);
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
