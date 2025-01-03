#include "DDH.hpp"

// one dimensional size of each domain decomp block. Each block has
// DDH_BLOCK_SIZE * DDH_BLOCK_SIZE degrees of freedom.
#define DDH_BLOCK_SIZE 16

using namespace cuddh;

WaveHoltz cuddh::init_waveholtz(double omega, double dt)
{
    WaveHoltz W;
    W.omega = omega;

    double T = (2 * M_PI) / omega;
    W.nt = std::ceil(T / dt);
    W.dt = T / W.nt;

    W.K.resize(W.nt+1);
    auto K = reshape(W.K.host_write(), W.nt+1);

    double tan_omega_dt = std::tan(0.5 * omega * dt);
    double a0 = 0.25 * (1 - tan_omega_dt * tan_omega_dt); // corrected shift

    for (int j = 0; j <= W.nt; ++j)
        K(j) = dt * (omega / M_PI) * (std::cos(omega * j * dt) - a0);
    K(0) *= 0.5;
    K(W.nt) *= 0.5;
    
    W.cs.resize(2 * W.nt + 1);
    W.sn.resize(2 * W.nt + 1);

    auto cs = reshape(W.cs.host_write(), 2 * W.nt + 1);
    auto sn = reshape(W.sn.host_write(), 2 * W.nt + 1);

    for (int j = 0; j <= 2 * W.nt; ++j)
    {
        double t = 0.5 * j * dt;
        cs(j) = -std::cos(omega * t);
        sn(j) = std::sin(omega * t);
    }

    return W;
}

template <typename Map, typename Key>
inline static bool contains(const Map &map, Key key)
{
    return map.find(key) != map.end();
}

template <int NB>
__device__ __forceinline__ static void stiffness(const int k,
                                                 const int l,
                                                 const int el,
                                                 const float3 G,
                                                 const int s_I[][NB][NB],
                                                 const float s_D[NB][NB],
                                                 float *const __restrict__ s_w, /* <- work */
                                                 float *const __restrict__ s_u, /* input and work */
                                                 float *const __restrict__ s_out)
{
    // compute covariant derivatives
    float Ux = 0.0f, Uy = 0.0f;
    int idx;

#pragma unroll NB
    for (int i = 0; i < NB; ++i)
    {
        idx = s_I[el][l][i];
        Ux += s_D[k][i] * s_u[idx];

        idx = s_I[el][i][k];
        Uy += s_D[l][i] * s_u[idx];
    }

    __syncthreads();

    // compute contravariant flux
    idx = k + NB * (l + NB * el);
    s_u[idx] = G.x * Ux + G.y * Uy;
    s_w[idx] = G.y * Ux + G.z * Uy;

    __syncthreads();

    // integrate against grad phi
    float Su = 0.0f;

#pragma unroll NB
    for (int i = 0; i < NB; ++i)
    {
        idx = i + NB * (l + NB * el);
        Su += s_D[i][k] * s_u[idx];

        idx = k + NB * (i + NB * el);
        Su += s_D[i][l] * s_w[idx];
    }

    idx = s_I[el][l][k];
    atomicAdd(s_out + idx, Su);
}

template <int NB, int NEL>
static void ddh_action(const EnsembleSpace *efem,
                       const int g_ndof,                          /* global finite element degrees of freedom */
                       const int n_domains,                       /* number of subdomains */
                       const int n_lambda,                        /* number of substructured DOFs (lambda) */
                       const int nt,                              /* number of time steps */
                       const float omega,                         /* Helmholtz frequency */
                       const float dt,                            /* time step */
                       const const_icube_wrapper B,               /* global lambda indices associated with boundary DOF */
                       const const_imat_wrapper gI,               /* global solution DOF associated with subdomain DOF */
                       const TensorWrapper<4, const int> sI,      /* mapping from (i,j)-node on element to subspace DOF */
                       const MatrixWrapper<const float> D,        /* differentiation matrix */
                       const MatrixWrapper<const float3> g,       /* geometric factors for stiffness computation */
                       const MatrixWrapper<const float> m,        /* subdomain mass matrices */
                       const MatrixWrapper<const float> g_inv_m,  /* global inverse mass matrix coefficients (mapped to subdomain index) */
                       const MatrixWrapper<const float> a,        /* variable coefficient */
                       const MatrixWrapper<const float> H,        /* subdomain boundary mass matrices */
                       const float *const __restrict__ wh_filter, /* WaveHoltz filter (cos(omega*t) - 1/4)*dt */
                       const float *const __restrict__ cs,        /* cos(omega*t) on half time steps */
                       const float *const __restrict__ sn,        /* sin(omega*t) on half time steps */
                       const double *const __restrict__ x,        /* input */
                       double *const __restrict__ y,              /* output */
                       const float *const __restrict__ d_lambda,  /* substructured problem variables */
                       float *const __restrict__ d_update /* substructured problem variables */)
{
    constexpr int wh_maxit = 5;

    auto s_dof = efem->sizes(MemorySpace::DEVICE);   // number of subdomain degrees of freedom
    auto s_fdof = efem->fsizes(MemorySpace::DEVICE); // number of face space degrees of freedom

    const float half_dt = 0.5f * dt;
    const float rw = 1.0f / omega;

    if (y)
        zeros(2 * g_ndof, y);

    const float *g_lambda = (d_lambda) ? d_lambda : nullptr;
    const float *g_mu = (d_lambda) ? (d_lambda + n_lambda) : nullptr;

    float *lambda_update = (d_update) ? d_update : nullptr;
    float *mu_update = (d_update) ? (d_update + n_lambda) : nullptr;

    constexpr int MX_NDOF = NB * NB * NEL * NEL; // == DDH_BLOCK_SIZE^2

    forall_1d(MX_NDOF, n_domains, [=] __device__(const int subsp) mutable -> void
              {
        const int tid = threadIdx.x; // thread id
        
        // get subspace dimensions
        const int fdof = s_fdof(subsp); // dimension of facespace
        const int ndof = s_dof(subsp); // dimension of subspace

#ifdef CUDDH_DEBUG
        assert(ndof <= MX_NDOF);
#endif

        // shared mem
        __shared__ float s_p_half[MX_NDOF];
        __shared__ float s_q_half[MX_NDOF];
        __shared__ float s_z[MX_NDOF];
        __shared__ float s_D[NB][NB];
        __shared__ int s_I[NEL*NEL][NB][NB];

        // convinient indicies
        const int k = tid % NB;
        const int l = (tid % (NB * NB)) / NB;
        const int el = tid / (NB * NB);

        // copy D
        if (tid < NB * NB)
            s_D[k][l] = D(k, l);

        // copy sI
        s_I[el][l][k] = sI(k, l, el, subsp);

        int g_idx = -1; // subspace DOF[tid] global index.
        
        float ai = 0.0f; // variable coefficient a(x)
        float mi = 0.0f; // subdomain mass matrix coefficient
        float inv_mi = 0.0f; // subdomain weighted inverse mass matrix coefficient
        
        float Hi = 0.0f; // subdomain boundary face mass matrix
        
        float F = 0.0f, G = 0.0f; // Helmholtz forcing
        float u = 0.0f, v = 0.0f; // (u,v) are the approx solution of the Helmholtz eq.
        float p = 0.0f, q = 0.0f; // (p,q) are the solution of the wave eq.
        float lambda = 0.0f, mu = 0.0f; // (lambda, mu) are the variables of the substructured problem.

        const float3 g_tid = g(tid, subsp);

        // copy global x to forcing, init work variables
        if (tid < ndof)
        {
            g_idx = gI(tid, subsp);
            ai = a(tid, subsp);
            mi = m(tid, subsp);
            inv_mi = 1.0f / (ai * ai * mi);

            if (x)
            {
                F = x[g_idx];
                G = x[g_ndof + g_idx];
            }
        }

        // add lambda to forcing, init work variables
        if (tid < fdof)
        {
            Hi = H(tid, subsp);

            if (d_lambda)
            {
                const int idx = B(tid, 0, subsp);
                if (idx >= 0)
                {
                    lambda = g_lambda[idx];
                    mu = g_mu[idx];

                    F += Hi * lambda;
                    G += Hi * mu;
                }
            }

            Hi *= ai;
        }

        // WaveHoltz iteration
        for (int whit=0; whit < wh_maxit; ++whit)
        {
            float dK = wh_filter[0];
            p = u;
            q = v;

            u *= dK;
            v *= dK;

            // time stepping
            for (int it=1; it <= nt; ++it)
            {
                // to save shared memory we use s_p_half and s_q_half as work
                // variables in the computation of the stiffness action. So
                // we copy p to s_p_half and pass s_p_half to stifness which
                // will compute the action on s_p_half but also overwrite it
                // with junk.

                s_z[tid] = 0.0f;
                s_p_half[tid] = p;
                __syncthreads();

                // z <- z + S * p, overwrites p and uses s_q_half as a work variable
                stiffness(k, l, el, g_tid, s_I, s_D, s_q_half, s_p_half, s_z);
                __syncthreads();

                s_z[tid] -= Hi * q;

                // half time step
                float dq = s_z[tid] + cs[2*it-2] * F;
                dq += sn[2*it-2] * G;
                dq *= inv_mi;
                
                s_p_half[tid] = p - half_dt * q;
                s_q_half[tid] = q + half_dt * dq;

                s_z[tid] = 0.0f;
                p -= dt * s_q_half[tid]; // <- full time step

                s_z[tid] -= Hi * s_q_half[tid];
                __syncthreads();

                stiffness(k, l, el, g_tid, s_I, s_D, s_q_half, s_p_half, s_z);
                __syncthreads();

                // full time step + WaveHoltz update
                dq = s_z[tid] + cs[2*it-1] * F;
                dq += sn[2*it-1] * G;
                dq *= inv_mi;

                q += dt * dq;

                dK = wh_filter[it];
                u += dK * p;
                v += dK * q;
            } // time stepping
        } // WaveHoltz

        // rescale v and update global solution
        v *= rw;

        if (y && (tid < ndof))
        {
            const float M = mi * g_inv_m(tid, subsp);
            
            const double m_u = M * u;
            atomicAdd(y+g_idx, m_u);

            const double m_v = M * v;
            atomicAdd(y+g_ndof+g_idx, m_v);
        }

        // update Lambdas
        if (d_update && (tid < fdof))
        {
            const int idx = B(tid, 1, subsp);
            if (idx >= 0)
            {
                const float S = 2.0f * ai * omega;
                lambda_update[idx] = -lambda - S * v;
                mu_update[idx]     = -mu     + S * u;
            }
        } });
}

static int partition_structured_mesh(std::unique_ptr<EnsembleSpace> &efem, const H1Space &fem, int nx, int ny, int n_basis)
{
    const int elems_per_domain_x = DDH_BLOCK_SIZE / n_basis;

    if (nx % elems_per_domain_x != 0 || ny % elems_per_domain_x != 0)
        cuddh_error("Only nx x ny meshes with nx and ny multiples of 32 / n_basis allowed.");

    const int num_domains_x = nx / elems_per_domain_x;
    const int num_domains_y = ny / elems_per_domain_x;

    int n_domains = num_domains_x * num_domains_y;

    imat element_labels(nx, ny);
    std::fill(element_labels.begin(), element_labels.end(), -1);

    for (int j = 0; j < ny; ++j)
    {
        for (int i = 0; i < nx; ++i)
        {
            int label_x = i / elems_per_domain_x;
            int label_y = j / elems_per_domain_x;
            element_labels(i, j) = label_x + num_domains_x * label_y;
        }
    }

    efem.reset(new EnsembleSpace(fem, n_domains, element_labels));

    return n_domains;
}

/**
 * @brief computes permuation of indices to natural ordering (face DOFs first).
 * 
 * @param h_gI Host array of size mx_dof * n_domains. gI maps subspace DOFs to global DOFs
 * @param h_sI Host array of size n_basis * n_basis * mx_elem * n_domains. sI maps local element DOFs to subspace DOFs
 * @param efem 
 */
static void natural_ordering(int *h_gI, int *h_sI, const EnsembleSpace *efem)
{
    const int n_domains = efem->size();
    const int mx_dof = efem->max_size();
    const int mx_elem = efem->max_n_elem();

    imat perm(mx_dof, n_domains);
    imat inv_perm(mx_dof, n_domains);

    auto P = efem->face_proj(MemorySpace::HOST);
    auto sizes = efem->sizes(MemorySpace::HOST);
    auto fsizes = efem->fsizes(MemorySpace::HOST);

    for (int subsp = 0; subsp < n_domains; ++subsp)
    {
        const int ndof = sizes(subsp);
        const int fdof = fsizes(subsp);

        for (int i = 0; i < ndof; ++i)
        {
            perm(i, subsp) = -1;
            inv_perm(i, subsp) = -1;
        }

        std::unordered_set<int> pp;
        int l = 0;
        for (; l < fdof; ++l)
        {
            int j = P(l, subsp);

            pp.insert(j);
            perm(l, subsp) = j;
        }

        for (int i = 0; i < ndof; ++i)
        {
            if (contains(pp, i))
                continue;

            perm(l, subsp) = i;
            ++l;
        }

        for (int i = 0; i < ndof; ++i)
        {
            int j = perm(i, subsp);
            inv_perm(j, subsp) = i;
        }
    }

    auto h_n_elems = efem->n_elems(MemorySpace::HOST);

    auto g_inds = efem->global_indices(MemorySpace::HOST);
    auto s_inds = efem->subspace_indices(MemorySpace::HOST);
    const int n_basis = s_inds.shape(0);

    auto gI = reshape(h_gI, mx_dof, n_domains);
    auto sI = reshape(h_sI, n_basis, n_basis, mx_elem, n_domains);

    std::fill(gI.begin(), gI.end(), -1);
    std::fill(sI.begin(), sI.end(), -1);

    for (int subsp = 0; subsp < n_domains; ++subsp)
    {
        const int ndof = sizes[subsp];

        for (int i = 0; i < ndof; ++i)
            gI(i, subsp) = g_inds(perm(i, subsp), subsp);

        const int nel = h_n_elems[subsp];
        for (int el = 0; el < nel; ++el)
        {
            for (int l = 0; l < n_basis; ++l)
            {
                for (int k = 0; k < n_basis; ++k)
                {
                    sI(k, l, el, subsp) = inv_perm(s_inds(k, l, el, subsp), subsp);
                }
            }
        }
    }
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

static void diffmat(float *h_D, const Basis &basis)
{
    const int n_basis = basis.size();
    dmat D(n_basis, n_basis);
    basis.deriv(n_basis, basis.quadrature().x(), D);
    for (int i = 0; i < n_basis * n_basis; ++i)
        h_D[i] = D[i];
}

static void DD_mass(float *h_m, const H1Space &fem, const EnsembleSpace *efem, const TensorWrapper<4, const int> &sI)
{
    const Mesh2D &mesh = fem.mesh();
    const Basis &basis = fem.basis();
    const QuadratureRule& q = basis.quadrature();

    const int n_basis = basis.size();
    const int n_domains = efem->size();
    const int mx_elem_per_dom = efem->max_n_elem();
    const int mx_dof = efem->max_size();

    auto h_n_elems = efem->n_elems(MemorySpace::HOST);
    auto h_elems = efem->elements(MemorySpace::HOST);

    const double *h_detJ = mesh.element_metrics(q).measures(MemorySpace::HOST);
    auto detJ = reshape(h_detJ, n_basis, n_basis, mesh.n_elem());

    auto m = reshape(h_m, mx_dof, n_domains);

    for (int subsp = 0; subsp < n_domains; ++subsp)
    {
        const int s_nel = h_n_elems(subsp);
        for (int el = 0; el < s_nel; ++el)
        {
            const int g_el = h_elems(el, subsp);
            for (int j = 0; j < n_basis; ++j)
            {
                for (int i = 0; i < n_basis; ++i)
                {
                    int l = sI(i, j, el, subsp);
                    m(l, subsp) += q.w(i) * q.w(j) * detJ(i, j, g_el);
                }
            }
        }
    }
}

static void DD_face_mass(float *h_h, const H1Space &fem, const EnsembleSpace *efem)
{
    const Mesh2D &mesh = fem.mesh();
    const Basis &basis = fem.basis();
    const QuadratureRule &q = basis.quadrature();

    const int n_domains = efem->size();
    const int n_basis = basis.size();
    const int mx_fdof = efem->max_fsize();

    auto n_faces = efem->n_faces(MemorySpace::HOST);
    auto faces = efem->faces(MemorySpace::HOST);
    auto f_inds = efem->face_indices(MemorySpace::HOST);

    auto H = reshape(h_h, mx_fdof, n_domains);

    for (int subsp = 0; subsp < n_domains; ++subsp)
    {
        const int s_nf = n_faces(subsp);
        for (int f = 0; f < s_nf; ++f)
        {
            const int g_f = faces(f, subsp);
            const Edge *edge = mesh.edge(g_f);

            for (int i = 0; i < n_basis; ++i)
            {
                const double xi = q.x(i);
                const double ds = edge->measure(xi);

                const int l = f_inds(i, f, subsp);
                H(l, subsp) += ds * q.w(i);
            }
        }
    }
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
static void DD_gridfun(T1 *h_u_dd, const T2 *h_u_mesh, const EnsembleSpace *efem, const TensorWrapper<2, const int> &gI)
{
    const int n_domains = efem->size();
    const int mx_dof = efem->max_size();
    
    auto sizes = efem->sizes(MemorySpace::HOST);

    auto dd = reshape(h_u_dd, mx_dof, n_domains);

    for (int subsp = 0; subsp < n_domains; ++subsp)
    {
        const int ndof = sizes(subsp);
        for (int i = 0; i < ndof; ++i)
            dd(i, subsp) = h_u_mesh[gI(i, subsp)];
    }
}

static void DD_geom_factors(float3 * d_G, const H1Space &fem, const EnsembleSpace *efem)
{
    const Mesh2D &mesh = fem.mesh();
    const Basis &basis = fem.basis();
    const QuadratureRule &q = basis.quadrature();

    const int n_basis = basis.size();
    const int n_domains = efem->size();
    const int mx_elem = efem->max_n_elem();

    host_device_dvec _w(n_basis);
    double *h_w = _w.host_write();
    for (int i = 0; i < n_basis; ++i)
        h_w[i] = q.w(i);
    auto w = reshape(_w.device_read(), n_basis);

    auto &metrics = mesh.element_metrics(q);
    
    const double *d_J = metrics.jacobians(MemorySpace::DEVICE);
    auto J = reshape(d_J, 2, 2, n_basis, n_basis, mesh.n_elem());

    auto n_elems = efem->n_elems(MemorySpace::DEVICE);
    auto elems = efem->elements(MemorySpace::DEVICE);

    auto G = reshape(d_G, n_basis, n_basis, mx_elem, n_domains);

    int n = n_basis * n_basis * mx_elem;
    forall_1d(n, n_domains, [=] __device__(int subsp) mutable -> void
    {
        const int n_elem = n_elems[subsp];

        const int el = threadIdx.x / (n_basis * n_basis);
        const int j = (threadIdx.x % (n_basis * n_basis)) / n_basis;
        const int i = threadIdx.x % n_basis;

        if (el >= n_elem)
            return;

        const int g_el = elems(el, subsp);

        const double W = w(i) * w(j);
        const double Y_eta = J(1, 1, i, j, g_el);
        const double X_eta = J(0, 1, i, j, g_el);
        const double Y_xi  = J(1, 0, i, j, g_el);
        const double X_xi  = J(0, 0, i, j, g_el);

        const double detJ = X_xi * Y_eta - X_eta * Y_xi;
        
        float3 gij;
        gij.x =  W * (Y_eta * Y_eta + X_eta * X_eta) / detJ;
        gij.y = -W * (Y_xi  * Y_eta + X_xi  * X_eta) / detJ;
        gij.z =  W * (Y_xi  * Y_xi  + X_xi  * X_xi)  / detJ;

        G(i, j, el, subsp) = gij;
    });
}

DDH::DDH(double omega, const double *h_a, const H1Space &fem, int nx, int ny)
    : g_ndof{fem.size()},
      g_elem{fem.mesh().n_elem()},
      n_basis{fem.basis().size()}
{
    // Setup domain decomposition by determining which elements belong to which
    // subdomain. For the structured meshes generated by Mesh2D::uniform_rect
    // this is straight forward.

    if (n_basis != 4 && n_basis != 8)
        cuddh_error("DDH error: Only n_basis==4, and n_basis==8 supported.");

    n_domains = partition_structured_mesh(efem, fem, nx, ny, n_basis);

    // determine max subspace dimensions
    mx_dof = efem->max_size();
    mx_fdof = efem->max_fsize();
    mx_elem_per_dom = efem->max_n_elem();

    // Set up lambdas for the substructured problem and compute index
    // information relating lambdas to their duals.
    _Bf.resize(2 * mx_fdof * n_domains);
    n_lambda = lambda_dofs(_Bf.host_write(), efem.get());

    // Permute global-to-subdomain and subspace-to-subdomain index mappings to natural ordering
    _gI.resize(mx_dof * n_domains);
    _sI.resize(n_basis * n_basis * mx_elem_per_dom * n_domains);
    natural_ordering(_gI.host_write(), _sI.host_write(), efem.get());

    auto sI = reshape(_sI.host_read(), n_basis, n_basis, mx_elem_per_dom, n_domains);
    auto gI = reshape(_gI.host_read(), mx_dof, n_domains);

    // Set up subspace finite element operators
    _D.resize(n_basis * n_basis);
    diffmat(_D.host_write(), fem.basis());

    _g_tensor.resize(n_basis * n_basis * mx_elem_per_dom * n_domains);
    DD_geom_factors(_g_tensor.device_write(), fem, efem.get());

    _m.resize(mx_dof * n_domains);
    DD_mass(_m.host_write(), fem, efem.get(), sI);

    _H.resize(mx_fdof * n_domains);
    DD_face_mass(_H.host_write(), fem, efem.get());

    _a.resize(mx_dof * n_domains);
    DD_gridfun(_a.host_write(), h_a, efem.get(), gI);
    const double amax = *std::max_element(h_a, h_a + g_ndof);

    dvec mi(fem.size());
    mass_matrix(mi.data(), fem);

    for (double& m : mi)
        m = 1.0 / m;

    _gmi.resize(mx_dof * n_domains);
    DD_gridfun(_gmi.host_write(), mi.data(), efem.get(), gI);

    // Setup WaveHoltz by determining the time step and precomputing the time
    // filter K(t) scaled by the quadrature weights (trapezoid rule), and also
    // the time dependency of the forcing (sines and cosines).
    double h = fem.mesh().min_h();
    double dt = 0.2 * 0.5 * h / (n_basis * n_basis); // TODO: incorporate amax
    W = init_waveholtz(omega, dt);
}

void DDH::action(const float *d_lambda, float *d_update) const
{
    auto B = reshape(_Bf.device_read(), mx_fdof, 2, n_domains);

    auto gI = reshape(_gI.device_read(), mx_dof, n_domains);
    auto sI = reshape(_sI.device_read(), n_basis, n_basis, mx_elem_per_dom, n_domains);

    auto D = reshape(_D.device_read(), n_basis, n_basis);

    auto g = reshape(_g_tensor.device_read(), n_basis * n_basis * mx_elem_per_dom, n_domains);

    auto a = reshape(_a.device_read(), mx_dof, n_domains);
    auto m = reshape(_m.device_read(), mx_dof, n_domains);
    auto H = reshape(_H.device_read(), mx_fdof, n_domains);
    auto g_inv_m = reshape(_gmi.device_read(), mx_dof, n_domains);

    // const float *wh_filter = _wh_filter.device_read();
    // const float *cs = _cs.device_read();
    // const float *sn = _sn.device_read();
    const float *wh_filter = W.K.device_read();
    const float *cs = W.cs.device_read();
    const float *sn = W.sn.device_read();

    if (n_basis == 4)
        ddh_action<4, DDH_BLOCK_SIZE / 4>(efem.get(), g_ndof, n_domains, n_lambda, W.nt, W.omega, W.dt, B, gI, sI, D, g, m, g_inv_m, a, H, wh_filter, cs, sn, (const double *)nullptr, (double *)nullptr, d_lambda, d_update);
    else if (n_basis == 8)
        ddh_action<8, DDH_BLOCK_SIZE / 8>(efem.get(), g_ndof, n_domains, n_lambda, W.nt, W.omega, W.dt, B, gI, sI, D, g, m, g_inv_m, a, H, wh_filter, cs, sn, (const double *)nullptr, (double *)nullptr, d_lambda, d_update);
    else
        cuddh_error("DDH::action only supports n_basis == 4 or 8.");

    axpby(2 * n_lambda, 1.0f, d_lambda, -1.0f, d_update);
}

void DDH::rhs(const double *f, float *b) const
{
    auto B = reshape(_Bf.device_read(), mx_fdof, 2, n_domains);

    auto gI = reshape(_gI.device_read(), mx_dof, n_domains);
    auto sI = reshape(_sI.device_read(), n_basis, n_basis, mx_elem_per_dom, n_domains);

    auto D = reshape(_D.device_read(), n_basis, n_basis);

    auto g = reshape(_g_tensor.device_read(), n_basis * n_basis * mx_elem_per_dom, n_domains);

    auto a = reshape(_a.device_read(), mx_dof, n_domains);
    auto m = reshape(_m.device_read(), mx_dof, n_domains);
    auto H = reshape(_H.device_read(), mx_fdof, n_domains);
    auto g_inv_m = reshape(_gmi.device_read(), mx_dof, n_domains);

    // const float *wh_filter = _wh_filter.device_read();
    // const float *cs = _cs.device_read();
    // const float *sn = _sn.device_read();
    const float *wh_filter = W.K.device_read();
    const float *cs = W.cs.device_read();
    const float *sn = W.sn.device_read();

    if (n_basis == 4)
        ddh_action<4, DDH_BLOCK_SIZE / 4>(efem.get(), g_ndof, n_domains, n_lambda, W.nt, W.omega, W.dt, B, gI, sI, D, g, m, g_inv_m, a, H, wh_filter, cs, sn, f, (double *)nullptr, (const float *)nullptr, b);
    else if (n_basis == 8)
        ddh_action<8, DDH_BLOCK_SIZE / 8>(efem.get(), g_ndof, n_domains, n_lambda, W.nt, W.omega, W.dt, B, gI, sI, D, g, m, g_inv_m, a, H, wh_filter, cs, sn, f, (double *)nullptr, (const float *)nullptr, b);
    else
        cuddh_error("DDH::action only supports n_basis == 4 or 8.");
}

void DDH::postprocess(const float *d_lambda, const double *f, double *y) const
{
    auto B = reshape(_Bf.device_read(), mx_fdof, 2, n_domains);

    auto gI = reshape(_gI.device_read(), mx_dof, n_domains);
    auto sI = reshape(_sI.device_read(), n_basis, n_basis, mx_elem_per_dom, n_domains);

    auto D = reshape(_D.device_read(), n_basis, n_basis);

    auto g = reshape(_g_tensor.device_read(), n_basis * n_basis * mx_elem_per_dom, n_domains);

    auto a = reshape(_a.device_read(), mx_dof, n_domains);
    auto m = reshape(_m.device_read(), mx_dof, n_domains);
    auto H = reshape(_H.device_read(), mx_fdof, n_domains);
    auto g_inv_m = reshape(_gmi.device_read(), mx_dof, n_domains);

    // const float *wh_filter = _wh_filter.device_read();
    // const float *cs = _cs.device_read();
    // const float *sn = _sn.device_read();
    const float *wh_filter = W.K.device_read();
    const float *cs = W.cs.device_read();
    const float *sn = W.sn.device_read();

    if (n_basis == 4)
        ddh_action<4, DDH_BLOCK_SIZE / 4>(efem.get(), g_ndof, n_domains, n_lambda, W.nt, W.omega, W.dt, B, gI, sI, D, g, m, g_inv_m, a, H, wh_filter, cs, sn, f, y, d_lambda, (float *)nullptr);
    else if (n_basis == 8)
        ddh_action<8, DDH_BLOCK_SIZE / 8>(efem.get(), g_ndof, n_domains, n_lambda, W.nt, W.omega, W.dt, B, gI, sI, D, g, m, g_inv_m, a, H, wh_filter, cs, sn, f, y, d_lambda, (float *)nullptr);
    else
        cuddh_error("DDH::action only supports n_basis == 4 or 8.");
}
