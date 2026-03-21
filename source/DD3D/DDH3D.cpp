#include "DD3D/DDH3D.hpp"

#include "FixedTensorWrapper.hpp"

#define DD3D_MX_DOF 512

using namespace cuddh;

namespace
{
    // Templated waveholtz used by host-side precomputation (make_alpha_beta).
    template <typename scalar_t>
    struct waveholtz_t
    {
        int nt;
        scalar_t omega;
        scalar_t dt;
        scalar_t weight;
        scalar_t shift;
        scalar_t theta;
        scalar_t sigma;

        constexpr __host__ __device__ scalar_t filter(scalar_t cs) const { return weight * cs - shift; }
    };

    template <typename scalar_t>
    static waveholtz_t<scalar_t> make_waveholtz_t(double omega, double dt)
    {
        waveholtz_t<scalar_t> W;
        W.omega = omega;

        double T = (2 * M_PI) / omega;
        W.nt = std::ceil(T / dt);
        dt = T / W.nt;
        W.dt = dt;

        double tan_omega_dt = std::tan(0.5 * omega * dt);
        double a0 = 0.25 * (1 - tan_omega_dt * tan_omega_dt);

        W.weight = 2.0 / W.nt;
        W.shift = W.weight * a0;
        W.theta = tan_omega_dt / omega;
        W.sigma = std::sin(0.5 * omega * dt) / (0.5 * omega);

        return W;
    }
} // namespace

// computes the complex multiplication (c + i*s) * (x + i*y) and stores the result in x and y.
__device__ __forceinline__ static void cxmult(float &x, float &y, float c, float s)
{
    float t = x;
    x = c * t - s * y;
    y = s * t + c * y;
}

template <int NB>
__device__ static constexpr int3 get_index3d(int i)
{
    int3 idx;
    idx.z = i / (NB * NB);
    idx.y = (i % (NB * NB)) / NB;
    idx.x = (i % (NB * NB)) % NB;

    return idx;
}

template <typename scalar_t, int NB, int NEL>
struct BlockStiffness
{
    using vec_t = cuddh::scalar3<scalar_t>;
    using mat_t = SmallSymmetricMatrix<scalar_t, 3>;

    struct SharedResources
    {
        scalar_t D[NB][NB];
        scalar_t u[NB * NB * NB * NEL];
        vec_t grad[NEL][NB][NB][NB];
    };

    SharedResources &smem;
    mat_t geom;
    int I[3][NB];

    __device__ BlockStiffness(SharedResources &mem, int subsp, int nel,
                              const DDStiffnessMatrix3D::DeviceDDStiffnessMatrix3D &stiffness_matrix,
                              const TensorWrapper<5, const int> &sI)
        : smem{mem}
    {
        const auto [x, y, z] = get_index3d<NB>(threadIdx.x);
        const auto el = threadIdx.y;

        if (el == 0 && z == 0)
            smem.D[x][y] = stiffness_matrix.D(x, y);

        for (int i = 0; i < NB; ++i)
        {
            I[0][i] = (el < nel) ? sI(i, y, z, el, subsp) : -1;
            I[1][i] = (el < nel) ? sI(x, i, z, el, subsp) : -1;
            I[2][i] = (el < nel) ? sI(x, y, i, el, subsp) : -1;
        }

        geom = (el < nel) ? stiffness_matrix.G(x, y, z, el, subsp) : mat_t{};

        __syncthreads();
    }

    __device__ scalar_t operator()(scalar_t in) const
    {
        const auto [x, y, z] = get_index3d<NB>(threadIdx.x);
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

/* implements ddh_action where each thread is resposible for a single DOF within an element and `EL_PER_THR` elements
 * per thread. Taking `EL_PER_THR > 1` sacrifices some parallelism in exchange for larger subdomains.
 */
template <typename scalar_t, int NB, int NEL>
static void ddh_action_dof_per_thread(
    const EnsembleSpace3D &efem, const int g_ndof, /* global finite element degrees of freedom */
    const int n_lambda,                            /* number of substructured DOFs (lambda) */
    const TensorWrapper<3, const int2> B,          /* global lambda indices associated with boundary DOFs */
    const TensorWrapper<3, const scalar_t> T,      /* lambda trace operator */
    const DDStiffnessMatrix3D::DeviceDDStiffnessMatrix3D stiffness_matrix,
    const MatrixWrapper<const cuddh::scalar2<scalar_t>> ab, /* alpha and beta coefficients for time-stepping */
    const MatrixWrapper<const scalar_t> punity,             /* partition of unity */
    const ::waveholtz_t<scalar_t> W,                        /* waveholtz data */
    const double *const __restrict__ d_x,                   /* input */
    double *const __restrict__ d_y,                         /* output */
    const scalar_t *const __restrict__ d_lambda,            /* substructured problem input DOFs */
    scalar_t *const __restrict__ d_update                   /* substructured problem output DOFs */
)
{
    using vec2 = cuddh::scalar2<scalar_t>;
    using vec3 = cuddh::scalar3<scalar_t>;
    using sym3x3 = SmallSymmetricMatrix<scalar_t, 3>;

    constexpr int wh_maxit = 20;

    constexpr int EDOF = NB * NB * NB;  // number of DOFs per element
    constexpr int MX_NDOF = EDOF * NEL; // maximum number of degrees of freedom per thread block

    const int n_domains = efem.size();

    auto gI = efem.global_indices(MemorySpace::DEVICE);   // global solution DOF associated with subdomain DOFs
    auto sI = efem.subspace_indices(MemorySpace::DEVICE); // mapping from (x,y,z)-node on element to subspace DOF

    auto subsp_elems = efem.n_elems(MemorySpace::DEVICE); // number of elements in a subdoamin
    auto subsp_sizes = efem.sizes(MemorySpace::DEVICE);   // number of subdomain degrees of freedom
    auto subsp_fsizes = efem.fsizes(MemorySpace::DEVICE); // number of face space degrees of freedom

    const scalar_t rw = 1 / W.omega;

    if (d_y)
        dla::zeros(2 * g_ndof, d_y);

    const scalar_t *g_lambda = (d_lambda) ? d_lambda : nullptr;
    const scalar_t *g_mu = (d_lambda) ? (d_lambda + n_lambda) : nullptr;

    scalar_t *lambda_update = (d_update) ? d_update : nullptr;
    scalar_t *mu_update = (d_update) ? (d_update + n_lambda) : nullptr;

    const scalar_t Rx = std::cos(0.5 * W.omega * W.dt);
    const scalar_t Ry = std::sin(0.5 * W.omega * W.dt);

    forall_2d(EDOF, NEL, n_domains, [=] __device__(const int subsp) mutable -> void {
        using BStiffness = BlockStiffness<scalar_t, NB, NEL>;

        const int tid = threadIdx.x + EDOF * threadIdx.y; // linearized thread index.

        // get subspace dimensions
        const int fdof = subsp_fsizes(subsp); // number of face space degrees of freedom
        const int ndof = subsp_sizes(subsp);  // number of subdomain degrees of freedom
        const int nel = subsp_elems(subsp);   // number of elements

        cuddh_assert(ndof <= MX_NDOF,
                     printf("DDH3D error: exceeded maximum number of subdomain DOFs per thread block (%d > %d)\n", ndof,
                            MX_NDOF););
        cuddh_assert(nel <= NEL, printf("DDH3D error: subdomain larger than block thread dimension can accomodate.\n"));

        __shared__ typename BStiffness::SharedResources smem;
        BStiffness A(smem, subsp, nel, stiffness_matrix, sI);

        const auto [alpha, beta] = [&]() -> vec2 {
            if (tid < ndof)
                return ab(tid, subsp);
            else
                return vec2{0, 0};
        }();

        const vec2 F = [&]() -> vec2 {
            vec2 F{0, 0};

            if (d_x && tid < ndof)
            {
                const int g_idx = gI(tid, subsp);
                const scalar_t weight = punity(tid, subsp);

                F.x = weight * d_x[g_idx];
                F.y = weight * d_x[g_ndof + g_idx];
            }

            if (d_lambda && tid < fdof)
            {
                for (int o = 0; o < 3; ++o)
                {
                    const int idx = B(o, tid, subsp).x;
                    if (idx >= 0)
                    {
                        F.x += g_lambda[idx];
                        F.y += g_mu[idx];
                    }
                }
            }

            return F;
        }();

        auto evolve_project = [&](vec2 u) -> vec2 {
            scalar_t cs = 1, sn = 0;
            scalar_t K = W.filter(cs);

            scalar_t p = u.x;

            cxmult(cs, sn, Rx, Ry);

            scalar_t q = (-sn * u.x + cs * u.y) * W.omega;

            u.x = K * p;

            K = W.filter(cs);
            u.y = K * q;

            for (int n = 1; n < W.nt; ++n)
            {
                cxmult(cs, sn, Rx, Ry);
                K = W.filter(cs);

                p += W.sigma * q;
                u.x += K * p;

                q = alpha * q + beta * (-A(p) + cs * F.x + sn * F.y);

                cxmult(cs, sn, Rx, Ry);
                K = W.filter(cs);

                u.y += K * q;
            }

            u.y *= rw;

            return u;
        };

        vec2 u{0, 0};
        for (int it = 0; it < wh_maxit; ++it)
        {
            u = evolve_project(u);
        }

        if (d_y && (tid < ndof))
        {
            const scalar_t weight = punity(tid, subsp);
            const int g_idx = gI(tid, subsp);

            const double m_u = weight * u.x;
            atomicAdd(d_y + g_idx, m_u);

            const double m_v = weight * u.y;
            atomicAdd(d_y + g_ndof + g_idx, m_v);
        }

        if (d_update && tid < fdof)
        {
            for (int o = 0; o < 3; ++o)
            {
                const auto [i, j] = B(o, tid, subsp);

                scalar_t lambda = 0, mu = 0;

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
        }
    });
}

// Populate _B (int2, leading dim 3) and _T (scalar_t, leading dim 3) from the connectivity map.
// B(o, i, p) = {lambda_index, dual_lambda_index} for face DOF i of subspace p, slot o (o in [0,3)).
// T(o, i, p) = 2 * omega * a(i, p) * dof.face_mass for the same DOF.
// Returns n_lambda = 2 * n_shared.
template <typename scalar_t>
static int lambda_dofs(thrust::universal_vector<int2> &h_B, thrust::universal_vector<scalar_t> &h_T,
                       const EnsembleSpace3D &efem, double omega, const MatrixWrapper<scalar_t> a)
{
    const int n_domains = efem.size();
    const int mx_fdof = efem.max_fsize();

    auto cmap = efem.connectivity_map(MemorySpace::HOST);
    const int n_shared = cmap.shape(0);
    const int n_lambda = 2 * n_shared;

    // leading dim 3: at most 3 coordinate-aligned faces can share a DOF in 3D
    h_B.resize(3 * mx_fdof * n_domains);
    h_T.resize(3 * mx_fdof * n_domains);
    thrust::fill(h_B.begin(), h_B.end(), int2{-1, -1});
    thrust::fill(h_T.begin(), h_T.end(), scalar_t(0));

    auto b = reshape(h_B, 3, mx_fdof, n_domains);
    auto t = reshape(h_T, 3, mx_fdof, n_domains);

    for (int k = 0; k < n_shared; ++k)
    {
        const LambdaDof dof = cmap(k);

        for (const int s : {0, 1})
        {
            const int subspace = dof.subspaces[s];
            const int face_index = dof.local_dof_indices[s];

            // find an available slot (o) for this DOF
            for (int o = 0; o < 3; ++o)
            {
                if (b(o, face_index, subspace).x < 0)
                {
                    b(o, face_index, subspace) = {(s == 0) ? k : n_shared + k, (s == 0) ? n_shared + k : k};
                    t(o, face_index, subspace) =
                        static_cast<scalar_t>(2.0 * omega * a(face_index, subspace) * dof.face_mass);
                    break;
                }
            }
        }
    }

    return n_lambda;
}

// Map global FEM function values to DD subspace arrays.
template <typename T1, typename T2>
static void DD_gridfun(T1 *h_u_dd, const T2 *h_u_mesh, const EnsembleSpace3D &efem)
{
    const int n_domains = efem.size();
    const int mx_dof = efem.max_size();

    auto sizes = efem.sizes(MemorySpace::HOST);
    auto gI = efem.global_indices(MemorySpace::HOST);

    auto dd = reshape(h_u_dd, mx_dof, n_domains);

    for (int subsp = 0; subsp < n_domains; ++subsp)
    {
        const int ndof = sizes(subsp);
        for (int i = 0; i < ndof; ++i)
            dd(i, subsp) = h_u_mesh[gI(i, subsp)];
    }
}

template <typename scalar_t>
static thrust::universal_vector<scalar_t> partition_of_unity(const H1Space3D &fem, const EnsembleSpace3D &efem)
{
    MassMatrix3D M(fem);
    DDMassMatrix3D DDM(fem, efem);

    auto d_m = diagonal_mass(M, MemorySpace::DEVICE);
    auto d_ddm = DDM.to_device();

    const int n_domains = efem.size();
    const int mx_dof = efem.max_size();

    auto sizes = efem.sizes(MemorySpace::DEVICE);
    auto gI = efem.global_indices(MemorySpace::DEVICE);

    thrust::universal_vector<scalar_t> P(mx_dof * n_domains, 0);
    auto p = reshape(P, mx_dof, n_domains);

    forall_1d(mx_dof, n_domains, [=] __device__(int subsp) mutable -> void {
        const int i = threadIdx.x;
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
                                                                          const H1Space3D &fem,
                                                                          const EnsembleSpace3D &efem)
{
    DDMassMatrix3D M(fem, efem);
    DDFaceMassMatrix3D H(fem, efem);

    auto m = M.to_device();
    auto h = H.to_device();

    const int n_domains = efem.size();
    const int mx_dof = efem.max_size();

    auto s_dof = efem.sizes(MemorySpace::DEVICE);
    auto s_fdof = efem.fsizes(MemorySpace::DEVICE);

    waveholtz_t<scalar_t> W = make_waveholtz_t<scalar_t>(omega, dt);

    thrust::universal_vector<cuddh::scalar2<scalar_t>> ab(mx_dof * n_domains);
    auto alpha_beta = reshape(ab, mx_dof, n_domains);

    forall_1d(mx_dof, n_domains, [=] __device__(int subsp) mutable -> void {
        const int i = threadIdx.x;
        const int ndof = s_dof(subsp);
        const int fdof = s_fdof(subsp);

        cuddh::scalar2<scalar_t> ab_i{0, 0};

        if (i < ndof)
        {
            scalar_t ai = a(i, subsp);
            scalar_t Mi = m(i, subsp);
            scalar_t Hi = (i < fdof) ? h(i, subsp) : scalar_t(0);

            Hi *= ai;
            Mi *= ai * ai;

            const scalar_t inv = 1 / (Mi + W.theta * Hi);
            ab_i.x = (Mi - W.theta * Hi) * inv;
            ab_i.y = W.sigma * inv;
        }

        alpha_beta(i, subsp) = ab_i;
    });

    CUDDH_CUDA_CHECK(cudaDeviceSynchronize());

    return ab;
}

template <typename scalar_t>
DDSubstructedProblem3D<scalar_t>::DDSubstructedProblem3D(double omega_, const double *h_a, const H1Space3D &fem,
                                                         const EnsembleSpace3D &efem_)
    : g_ndof{fem.size()},
      g_elem{fem.mesh().n_elem()},
      n_basis{fem.basis().size()},
      omega{omega_},
      efem{efem_},
      S(fem, efem_)
{
    cuddh_verify(n_basis >= 2 && n_basis <= 4, printf("DDH3D error: Only n_basis in [2,3,4] supported.\n"););
    cuddh_verify(efem.max_size() <= DD3D_MX_DOF, printf("DDH3D error: subdomains too big.\n"));

    n_domains = efem.size();
    mx_dof = efem.max_size();
    mx_fdof = efem.max_fsize();
    mx_elem_per_dom = efem.max_n_elem();

    thrust::universal_vector<scalar_t> _a(mx_dof * n_domains);
    auto a = reshape(_a, mx_dof, n_domains);
    DD_gridfun(a.data(), h_a, efem);

    n_lambda = lambda_dofs(_B, _T, efem, omega, a);

    _partition_of_unity = partition_of_unity<scalar_t>(fem, efem);

    // time step determined by CFL condition: dt = C * h / (n_basis * n_basis * max_vel)
    const double h = fem.mesh().h();
    const double reciprocal_max_vel = *std::min_element(h_a, h_a + g_ndof);
    dt = 2.0 * reciprocal_max_vel * h / (n_basis * n_basis);

    alpha_beta = make_alpha_beta<scalar_t>(omega, dt, a, fem, efem);
}

template <typename scalar_t>
void DDSubstructedProblem3D<scalar_t>::action(const double *fem_in, double *fem_out, const scalar_t *lambda_in,
                                              scalar_t *lambda_out) const
{
    constexpr int MX_NDOF = DD3D_MX_DOF;
    using func_t = decltype(&ddh_action_dof_per_thread<scalar_t, 2, MX_NDOF / (2 * 2 * 2)>);

    auto func = [&]() -> func_t {
        switch (n_basis)
        {
            case 2:
                return ddh_action_dof_per_thread<scalar_t, 2, MX_NDOF / (2 * 2 * 2)>;
            case 3:
                return ddh_action_dof_per_thread<scalar_t, 3, MX_NDOF / (3 * 3 * 3)>;
            case 4:
                return ddh_action_dof_per_thread<scalar_t, 4, MX_NDOF / (4 * 4 * 4)>;
            default:
                cuddh_verify(n_basis < 4, printf("DDH3D error: not implemented for n_basis > 4.\n"));
                return nullptr;
        }
    }();

    auto B = reshape(_B, 3, mx_fdof, n_domains);
    auto T = reshape(_T, 3, mx_fdof, n_domains);
    auto d_S = S.to_device();
    auto ab = reshape(alpha_beta, mx_dof, n_domains);
    auto punity = reshape(_partition_of_unity, mx_dof, n_domains);

    (*func)(efem, g_ndof, n_lambda, B, T, d_S, ab, punity, ::make_waveholtz_t<scalar_t>(omega, dt), fem_in, fem_out,
            lambda_in, lambda_out);
}

template <typename scalar_t>
void DDSubstructedProblem3D<scalar_t>::action(const scalar_t *x, scalar_t *y) const
{
    dla::zeros(2 * n_lambda, y);
    action((const double *)nullptr, (double *)nullptr, x, y);
    dla::axpby(2 * n_lambda, scalar_t(1), x, scalar_t(-1), y);
}

template <typename scalar_t>
void DDSubstructedProblem3D<scalar_t>::rhs(const double *f, scalar_t *b) const
{
    dla::zeros(2 * n_lambda, b);
    action(f, (double *)nullptr, (const scalar_t *)nullptr, b);
}

template <typename scalar_t>
void DDSubstructedProblem3D<scalar_t>::postprocess(const scalar_t *lambda, const double *f, double *y) const
{
    action(f, y, lambda, (scalar_t *)nullptr);
}

template <typename scalar_t>
void DDSubstructedProblem3D<scalar_t>::residual(const double *u, const double *f, double *res) const
{
    cuddh_verify(false, printf("DDSubstructedProblem3D::residual not yet implemented\n"));
}

namespace cuddh
{
    template class DDSubstructedProblem3D<float>;
    template class DDH3D<float>;
} // namespace cuddh
