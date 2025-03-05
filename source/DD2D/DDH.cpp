#include "DD2D/DDH.hpp"

// one dimensional size of each domain decomp block. Each block has
// DDH_BLOCK_SIZE * DDH_BLOCK_SIZE degrees of freedom.
#define DDH_BLOCK_SIZE 16

using namespace cuddh;

template <typename Map, typename Key>
inline static bool contains(const Map &map, Key key)
{
    return map.find(key) != map.end();
}

// computes the complex multiplication (c + i*s) * (x + i*y) and stores the result in x and y.
__device__ __forceinline__ static void cxmult(float &x, float &y, float c, float s)
{
    float t = x;
    x = c * t - s * y;
    y = s * t + c * y;
}

template <int NB>
__device__ __forceinline__ static void stiffness_matvec(const float3 geom,
                                                 const int Ix[NB],
                                                 const int Iy[NB],
                                                 const float s_D[NB][NB],
                                                 float2 s_w[][NB][NB], /* <- work */
                                                 float *const __restrict__ s_u, /* input */
                                                 float *const __restrict__ s_out)
{
    const auto [k, l, el] = threadIdx;

    float2 grad{0.0f, 0.0f};

    #pragma unroll NB
    for (int i = 0; i < NB; ++i)
        grad.x += s_D[k][i] * s_u[Ix[i]];

    #pragma unroll NB
    for (int i = 0; i < NB; ++i)
        grad.y += s_D[l][i] * s_u[Iy[i]];

    s_w[el][l][k].x = geom.x * grad.x + geom.y * grad.y;
    s_w[el][l][k].y = geom.y * grad.x + geom.z * grad.y;
    __syncthreads();

    float Su = 0.0f;

    #pragma unroll NB
    for (int i = 0; i < NB; ++i)
        Su += s_D[i][k] * s_w[el][l][i].x;

    #pragma unroll NB
    for (int i = 0; i < NB; ++i)
        Su += s_D[i][l] * s_w[el][i][k].y;

    atomicAdd(s_out + Ix[k], Su);
    __syncthreads();
}

template <int NB, int NEL>
static void ddh_action(const EnsembleSpace *efem,
                       const int g_ndof,                          /* global finite element degrees of freedom */
                       const int n_domains,                       /* number of subdomains */
                       const int n_lambda,                        /* number of substructured DOFs (lambda) */
                       const const_icube_wrapper B,               /* global lambda indices associated with boundary DOF */
                       const const_imat_wrapper gI,               /* global solution DOF associated with subdomain DOF */
                       const TensorWrapper<4, const int> sI,      /* mapping from (i,j)-node on element to subspace DOF */
                       const DDStiffnessMatrix::DeviceDDStiffnessMatrix stiffness_matrix,/* stiffness_matvec matrix on device */
                       const MatrixWrapper<const float> m,        /* subdomain mass matrices */
                       const MatrixWrapper<const float> g_inv_m,  /* global inverse mass matrix coefficients (mapped to subdomain index) */
                       const MatrixWrapper<const float> a,        /* variable coefficient */
                       const MatrixWrapper<const float> H,        /* subdomain boundary mass matrices */
                       const WaveHoltz W,                         /* WaveHoltz data */
                       const double *const __restrict__ x,        /* input */
                       double *const __restrict__ y,              /* output */
                       const float *const __restrict__ d_lambda,  /* substructured problem variables */
                       float *const __restrict__ d_update /* substructured problem variables */)
{
    constexpr int wh_maxit = 5;

    auto s_dof = efem->sizes(MemorySpace::DEVICE);   // number of subdomain degrees of freedom
    auto s_fdof = efem->fsizes(MemorySpace::DEVICE); // number of face space degrees of freedom

    const float dt = W.dt;
    const float rw = 1.0f / W.omega;

    if (y)
        zeros(2 * g_ndof, y);

    const float *g_lambda = (d_lambda) ? d_lambda : nullptr;
    const float *g_mu = (d_lambda) ? (d_lambda + n_lambda) : nullptr;

    float *lambda_update = (d_update) ? d_update : nullptr;
    float *mu_update = (d_update) ? (d_update + n_lambda) : nullptr;

    constexpr int MX_NDOF = NB * NB * NEL * NEL; // == DDH_BLOCK_SIZE^2

    // R = Rx + i Ry = exp(i * omega * dt) used in the short term recurrence: z(t + dt) = R * z(t) where z(t) = exp(i * omega * t).
    const float Rx = std::cos(W.omega * W.dt);
    const float Ry = std::sin(W.omega * W.dt);

    forall_3d(NB, NB, NEL*NEL, n_domains, [=] __device__ (const int subsp) mutable -> void
    {
        const auto [k, l, el] = threadIdx; // convinient indicies
        const int tid = threadIdx.x + NB * (threadIdx.y + NB * threadIdx.z); // linearized thread id
        
        // get subspace dimensions
        const int fdof = s_fdof(subsp); // dimension of facespace
        const int ndof = s_dof(subsp); // dimension of subspace

#ifdef CUDDH_DEBUG
        assert(ndof <= MX_NDOF);
#endif

        // shared mem
        __shared__ float s_p[MX_NDOF];
        __shared__ float2 s_work[NEL*NEL][NB][NB];
        __shared__ float s_Sp[MX_NDOF];
        __shared__ float s_D[NB][NB];

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

        float Hi = 0.0f; // subdomain boundary face mass matrix

        float Q0 = 0.0f; // used for time-stepping
        float Q1 = 0.0f;
        float delta = 0.0f;
        
        float F = 0.0f, G = 0.0f; // Helmholtz forcing
        float u = 0.0f, v = 0.0f; // (u,v) are the approx solution of the Helmholtz eq.
        float p = 0.0f, q = 0.0f; // (p,q) are the solution of the wave eq.

        const float3 geom = stiffness_matrix.G(tid, subsp);

        if (tid < ndof)
        {
            // copy global x to forcing
            if (x)
            {
                const int g_idx = gI(tid, subsp);

                F = x[g_idx];
                G = x[g_ndof + g_idx];
            }

            const float ai = a(tid, subsp); // variable coefficient

            if (tid < fdof)
            {
                Hi = H(tid, subsp);

                // add lambda to forcing
                const int idx = B(tid, 0, subsp);
                if (d_lambda && idx >= 0)
                {
                    F += Hi * g_lambda[idx];
                    G += Hi * g_mu[idx];
                }

                Hi *= ai;
            }

            Q0 = ai * ai * m(tid, subsp); // used for time stepping
            Q1 = 1.0f / (Q0 + 0.5f * dt * Hi);
            delta = 0.5f * dt * dt / Q0;

            Q0 = Q0 * Q1 - 1.0f;
            Q1 *= 0.5f * dt;
        }

        // returns S * x where S is the stiffness matrix
        auto S = [&](float x) -> float
        {
            s_Sp[tid] = 0.0f;
            s_p[tid] = x;
            __syncthreads();

            stiffness_matvec(geom, Ix, Iy, s_D, s_work, s_p, s_Sp);

            return s_Sp[tid];
        };

        // WaveHoltz iteration
        for (int whit=0; whit < wh_maxit; ++whit)
        {
            float cs = 1.0f;
            float sn = 0.0f;

            float K = W.K(cs);
            
            p = u;
            q = v;

            u *= K;
            v *= K;

            // compute acceleration at t == 0
            float acc_n = S(p) - Hi * q - F * cs + G * sn;

            // time stepping
            for (int it=1; it < W.nt; ++it)
            {
                // update p
                p -= dt * q + delta * acc_n;
                
                cxmult(cs, sn, Rx, Ry); // update cos and sin
                float acc_n1 = S(p) - F * cs + G * sn;

                // update q
                q += Q0 * q + Q1 * acc_n + Q1 * acc_n1;

                // update acc
                acc_n = acc_n1 - Hi * q;

                // waveholtz update
                K = W.K(cs);
                u += K * p;
                v += K * q;
            } // time stepping
        } // WaveHoltz

        // rescale v and update global solution
        v *= rw;

        if (y && (tid < ndof))
        {
            const float M = m(tid, subsp) * g_inv_m(tid, subsp);
            const int g_idx = gI(tid, subsp);
            
            const double m_u = M * u;
            atomicAdd(y+g_idx, m_u);

            const double m_v = M * v;
            atomicAdd(y+g_ndof+g_idx, m_v);
        }

        // update Lambdas
        if (d_update && (tid < fdof))
        {
            const int i = B(tid, 0, subsp);
            float lambda = 0.0f;
            float mu = 0.0f;

            if (d_lambda && i >= 0)
            {
                lambda = g_lambda[i];
                mu = g_mu[i];
            }
            
            const int j = B(tid, 1, subsp);
            if (j >= 0)
            {
                const float T = 2.0f * a(tid, subsp) * W.omega;
                lambda_update[j] = -lambda - T * v;
                mu_update[j]     = -mu     + T * u;
            }
        }
    });
}

/**
 * Lambdas are ordered like: lambda = (lambda0, lambda1, mu0, mu1) where lambda0 is the "interior"
 * trace for each subspace and lambda1 is the "external" trace. cmap is a one-to-one map
 * between lambda1 and lambda2 in the respective face spaces of each subspace, so we assign
 * the cmap(0, k) to lambda0, and take the index of lambda1 to be that of lambda0 offset by
 * n_shared. Consequently, the indices of mu0 and mu1 are those of lambda0 and lambda1,
 * respectively, offset by n_lambda (=2*n_shared).
 * 
 * B is a 3D array of size (mx_fdof, 2, n_domains) where B(f, s, p) is the index in lambda of
 * face degree of freedom f for subspace p, and s is 0 for "interior" trace and 1 for "external" trace.
 */
static int lambda_dofs(int *h_B, const EnsembleSpace *efem)
{
    const int n_domains = efem->size();
    const int mx_fdof = efem->max_fsize();
    
    auto cmap = efem->connectivity_map(MemorySpace::HOST);
    const int n_shared = cmap.shape(1);
    const int n_lambda = 2 * n_shared;

    auto B = reshape(h_B, mx_fdof, 2, n_domains);
    std::fill(B.begin(), B.end(), -1);

    for (int k = 0; k < n_shared; ++k)
    {
        int subspace0 = cmap(0, k);
        int subspace1 = cmap(1, k);
        int face_index0 = cmap(2, k);
        int face_index1 = cmap(3, k);

        B(face_index0, 0, subspace0) = k;
        B(face_index0, 1, subspace0) = n_shared + k;
        B(face_index1, 0, subspace1) = n_shared + k;
        B(face_index1, 1, subspace1) = k;
    }

    return n_lambda;
}

static void mass_matrix(double *h_m, const H1Space &fem)
{
    const Mesh2D &mesh = fem.mesh();
    const Basis &basis = fem.basis();
    const QuadratureRule &q = basis.quadrature();

    const int g_elem = mesh.n_elem();
    const int n_basis = basis.size();

    const int g_ndof = fem.size();

    auto &metrics = mesh.element_metrics(q);
    const double *h_detJ = metrics.measures(MemorySpace::HOST);
    auto detJ = reshape(h_detJ, n_basis, n_basis, g_elem);

    auto inds = fem.global_indices(MemorySpace::HOST);

    auto m = reshape(h_m, g_ndof);
    
    for (int el = 0; el < g_elem; ++el)
    {
        for (int j = 0; j < n_basis; ++j)
        {
            for (int i = 0; i < n_basis; ++i)
            {
                int l = inds(i, j, el);
                m(l) += q.w(i) * q.w(j) * detJ(i, j, el);
            }
        }
    }
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

DDH::DDH(double omega, const double *h_a, const H1Space &fem, const EnsembleSpace &efem)
    : g_ndof{fem.size()},
      g_elem{fem.mesh().n_elem()},
      n_basis{fem.basis().size()},
      efem{efem},
      M(fem, efem),
      H(fem, efem),
      S(fem, efem)
{
    // Setup domain decomposition by determining which elements belong to which
    // subdomain. For the structured meshes generated by Mesh2D::uniform_rect
    // this is straight forward.

    if (n_basis != 4 && n_basis != 8)
        cuddh_error("DDH error: Only n_basis==4, and n_basis==8 supported.");

    n_domains = efem.size();

    // determine max subspace dimensions
    mx_dof = efem.max_size();
    mx_fdof = efem.max_fsize();
    mx_elem_per_dom = efem.max_n_elem();

    // Set up lambdas for the substructured problem and compute index
    // information relating lambdas to their duals.
    _Bf.resize(2 * mx_fdof * n_domains);
    n_lambda = lambda_dofs(_Bf.host_write(), &efem);

    _a.resize(mx_dof * n_domains);
    DD_gridfun(_a.host_write(), h_a, &efem);
    const double reciprocal_max_vel = *std::min_element(h_a, h_a + g_ndof);

    dvec mi(fem.size());
    mass_matrix(mi.data(), fem);

    for (double& m : mi)
        m = 1.0 / m;

    _gmi.resize(mx_dof * n_domains);
    DD_gridfun(_gmi.host_write(), mi.data(), &efem);

    // time step determined by CFL condition: dt = C * h / (n_basis * n_basis * max_vel)
    double h = fem.mesh().min_h();
    double dt = 2.6 * reciprocal_max_vel * h / (n_basis * n_basis); // why 2.6?
    W = init_waveholtz(omega, dt);
}

void DDH::action(const float *d_lambda, float *d_update) const
{
    auto B = reshape(_Bf.device_read(), mx_fdof, 2, n_domains);

    auto gI = efem.global_indices(MemorySpace::DEVICE);
    auto sI = efem.subspace_indices(MemorySpace::DEVICE);

    auto d_S = S.to_device();
    auto d_M = M.to_device();
    auto d_H = H.to_device();

    auto a = reshape(_a.device_read(), mx_dof, n_domains);
    auto g_inv_m = reshape(_gmi.device_read(), mx_dof, n_domains);

    if (n_basis == 4)
        ddh_action<4, DDH_BLOCK_SIZE / 4>(&efem, g_ndof, n_domains, n_lambda, B, gI, sI, d_S, d_M, g_inv_m, a, d_H, W, (const double *)nullptr, (double *)nullptr, d_lambda, d_update);
    else if (n_basis == 8)
        ddh_action<8, DDH_BLOCK_SIZE / 8>(&efem, g_ndof, n_domains, n_lambda, B, gI, sI, d_S, d_M, g_inv_m, a, d_H, W, (const double *)nullptr, (double *)nullptr, d_lambda, d_update);
    else
        cuddh_error("DDH::action only supports n_basis == 4 or 8.");

    axpby(2 * n_lambda, 1.0f, d_lambda, -1.0f, d_update);
}

void DDH::rhs(const double *f, float *b) const
{
    auto B = reshape(_Bf.device_read(), mx_fdof, 2, n_domains);

    auto gI = efem.global_indices(MemorySpace::DEVICE);
    auto sI = efem.subspace_indices(MemorySpace::DEVICE);

    auto d_S = S.to_device();
    auto d_M = M.to_device();
    auto d_H = H.to_device();

    auto a = reshape(_a.device_read(), mx_dof, n_domains);
    auto g_inv_m = reshape(_gmi.device_read(), mx_dof, n_domains);

    if (n_basis == 4)
        ddh_action<4, DDH_BLOCK_SIZE / 4>(&efem, g_ndof, n_domains, n_lambda, B, gI, sI, d_S, d_M, g_inv_m, a, d_H, W, f, (double *)nullptr, (const float *)nullptr, b);
    else if (n_basis == 8)
        ddh_action<8, DDH_BLOCK_SIZE / 8>(&efem, g_ndof, n_domains, n_lambda, B, gI, sI, d_S, d_M, g_inv_m, a, d_H, W, f, (double *)nullptr, (const float *)nullptr, b);
    else
        cuddh_error("DDH::action only supports n_basis == 4 or 8.");
}

void DDH::postprocess(const float *d_lambda, const double *f, double *y) const
{
    auto B = reshape(_Bf.device_read(), mx_fdof, 2, n_domains);

    auto gI = efem.global_indices(MemorySpace::DEVICE);
    auto sI = efem.subspace_indices(MemorySpace::DEVICE);

    auto d_S = S.to_device();
    auto d_M = M.to_device();
    auto d_H = H.to_device();

    auto a = reshape(_a.device_read(), mx_dof, n_domains);
    auto g_inv_m = reshape(_gmi.device_read(), mx_dof, n_domains);

    if (n_basis == 4)
        ddh_action<4, DDH_BLOCK_SIZE / 4>(&efem, g_ndof, n_domains, n_lambda, B, gI, sI, d_S, d_M, g_inv_m, a, d_H, W, f, y, d_lambda, (float *)nullptr);
    else if (n_basis == 8)
        ddh_action<8, DDH_BLOCK_SIZE / 8>(&efem, g_ndof, n_domains, n_lambda, B, gI, sI, d_S, d_M, g_inv_m, a, d_H, W, f, y, d_lambda, (float *)nullptr);
    else
        cuddh_error("DDH::action only supports n_basis == 4 or 8.");
}
