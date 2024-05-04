#include "DDH.hpp"

// one dimensional size of each domain decomp block. Each block has
// DDH_BLOCK_SIZE * DDH_BLOCK_SIZE degrees of freedom.
#define DDH_BLOCK_SIZE 16

using namespace cuddh;

template <typename Map, typename Key>
inline static bool contains(const Map & map, Key key)
{
    return map.find(key) != map.end();
}

static void init_geom_factors(int n_domains,
                              int mx_elems,
                              int g_elem,
                              int n_basis,
                              const_ivec_wrapper n_elems,
                              const_imat_wrapper elems,
                              const double * __restrict__ d_w,
                              const double * __restrict__ d_J,
                              float * __restrict__ d_G)
{
    auto w = reshape(d_w, n_basis);
    auto J = reshape(d_J, 2, 2, n_basis, n_basis, g_elem);

    auto G = reshape(d_G, 3, n_basis, n_basis, mx_elems, n_domains);

    int n = n_basis * n_basis * mx_elems;
    forall_1d(n, n_domains, [=] __device__ (int subsp) mutable -> void
    {
        const int n_elem = n_elems[subsp];

        const int el = threadIdx.x / (n_basis * n_basis);
        const int j = (threadIdx.x % (n_basis * n_basis)) / n_basis;
        const int i = threadIdx.x % n_basis;

        if (el >= n_elem) return;

        const int g_el = elems(el, subsp);

        const double W = w(i) * w(j);
        const double Y_eta = J(1, 1, i, j, g_el);
        const double X_eta = J(0, 1, i, j, g_el);
        const double Y_xi  = J(1, 0, i, j, g_el);
        const double X_xi  = J(0, 0, i, j, g_el);

        const double detJ = X_xi * Y_eta - X_eta * Y_xi;
        
        G(0, i, j, el, subsp) =  W * (Y_eta * Y_eta + X_eta * X_eta) / detJ;
        G(1, i, j, el, subsp) = -W * (Y_xi  * Y_eta + X_xi  * X_eta) / detJ;
        G(2, i, j, el, subsp) =  W * (Y_xi  * Y_xi  + X_xi  * X_xi)  / detJ;
    });
}

template <int NB>
__device__ static inline void stiffness(const int k,
                                        const int l,
                                        const int el,
                                        const float G[3],
                                        const int s_I[][NB][NB],
                                        const float s_D[NB][NB],
                                        float * const __restrict__ s_w, /* <- work */
                                        float * const __restrict__ s_u, /* input and work */
                                        float * const __restrict__ s_out)
{
    int idx;

    // compute covariant derivatives
    float Dx = 0.0, Dy = 0.0;
    
    #pragma unroll NB
    for (int i = 0; i < NB; ++i)
    {
        idx = s_I[el][l][i];
        Dx += s_D[k][i] * s_u[idx];
        
        idx = s_I[el][i][k];
        Dy += s_D[l][i] * s_u[idx];
    }

    __syncthreads();

    // compute contravariant flux 
    idx = k + NB*(l + NB*el);
    s_u[idx] = G[0] * Dx + G[1] * Dy;

    idx = k + NB*(l + NB*el);
    s_w[idx] = G[1] * Dx + G[2] * Dy;

    __syncthreads();

    // integrate against grad phi
    float Su = 0.0;
    
    #pragma unroll NB
    for (int i = 0; i < NB; ++i)
    {
        idx = i + NB*(l + NB * el);
        Su += s_D[i][k] * s_u[idx];

        idx = k + NB*(i + NB * el);
        Su += s_D[i][l] * s_w[idx];
    }
    
    idx = s_I[el][l][k];
    atomicAdd(s_out + idx, Su);
}

template <int NB, int NEL>
static void ddh_action(const EnsembleSpace * efem,
                       const int g_ndof,
                       const int n_domains,
                       const int n_lambda,
                       const int nt,
                       const float omega,
                       const float dt,
                       const const_icube_wrapper B,
                       const const_imat_wrapper gI,
                       const TensorWrapper<4,const int> sI,
                       const MatrixWrapper<const float> D,
                       const TensorWrapper<3, const float> g,
                       const MatrixWrapper<const float> m,
                       const MatrixWrapper<const float> g_inv_m,
                       const MatrixWrapper<const float> a,
                       const MatrixWrapper<const float> H,
                       const VectorWrapper<const float> wh_filter,
                       const VectorWrapper<const float> cs,
                       const VectorWrapper<const float> sn,
                       const double * const __restrict__ x,
                       double * const __restrict__ y,
                       float * const __restrict__ d_lambda,
                       float * const __restrict__ d_update)
{
    constexpr int lambda_maxit = 1;
    constexpr int wh_maxit = 2;

    auto s_dof = efem->sizes(MemorySpace::DEVICE);  // number of subdomain degrees of freedom
    auto s_fdof = efem->fsizes(MemorySpace::DEVICE); // number of face space degrees of freedom
    
    const float half_dt = 0.5f * dt;
    const float rw = 1.0f / omega;

    auto g_lambda = reshape(d_lambda, n_lambda);
    auto g_mu = reshape(d_lambda + n_lambda, n_lambda);

    auto lambda_update = reshape(d_update, n_lambda);
    auto mu_update = reshape(d_update + n_lambda, n_lambda);

    zeros(2 * n_lambda, d_lambda);

    constexpr int MX_NDOF = NB * NB * NEL * NEL; // == DDH_BLOCK_SIZE^2

    for (int lit = 0; lit < lambda_maxit; ++lit)
    {
        zeros(2 * g_ndof, y);

        forall_1d(MX_NDOF, n_domains, [=] __device__ (const int subsp) mutable -> void
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

            // each thread has an associated lambda DOF, subspace DOF, and
            // FaceSpace DOF. The following variables are indices of those DOFs
            // in various arrays. Additionally, the elements of the (diagonal)
            // mass matrices are also kept.

            int g_idx; // subspace DOF[tid] global index.
            
            float ai; // variable coefficient a(x)
            float mi; // subdomain mass matrix coefficient
            float inv_mi; // subdomain weighted inverse mass matrix coefficient
            
            float Hi; // subdomain boundary face mass matrix
            
            float F = 0.0f; // Helmholtz forcing
            float G = 0.0f;
            float u = 0.0f; // (u,v) are the approx solution of the Helmholtz eq.
            float v = 0.0f;
            float p = 0.0f; // (p,q) are the solution of the wave eq.
            float q = 0.0f;

            // (lambda, mu) are the variables of the substructured problem.
            // There are two of each because corners have two values associated
            // with the same degree of freedom (because two edges meet).
            float lambda, mu;

            const float g_tid[] = {g(0, tid, subsp),
                                   g(1, tid, subsp),
                                   g(2, tid, subsp)};

            // copy global x to forcing, init work variables
            if (tid < ndof)
            {
                g_idx = gI(tid, subsp);
                ai = a(tid, subsp);
                mi = m(tid, subsp);
                inv_mi = 1.0f / (ai * ai * mi);

                F = x[g_idx];
                G = x[g_ndof + g_idx];
            }

            // add lambda to forcing, init work variables
            if (tid < fdof)
            {
                Hi = H(tid, subsp);

                int idx = B(tid, 0, subsp);
                if (idx >= 0)
                {
                    lambda = g_lambda[idx];
                    mu = g_mu[idx];

                    F += Hi * lambda;
                    G += Hi * mu;
                }

                Hi *= ai;
            }

            // WaveHoltz iteration
            // p, q are the time dependant variables.
            // u, v are the solution estimates.
            for (int whit=0; whit < wh_maxit; ++whit)
            {
                __syncthreads();
                
                float dK = wh_filter[0]; // please be in L1
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

                    if (tid < ndof)
                    {
                        s_z[tid] = 0.0f;
                        s_p_half[tid] = p;
                    }
                    __syncthreads();

                    // z <- z + S * p, overwrites p and uses s_q_half as a work variable
                    stiffness(k, l, el, g_tid, s_I, s_D, s_q_half, s_p_half, s_z);
                    __syncthreads();

                    // z <- z - H * q
                    if (tid < fdof)
                        s_z[tid] -= Hi * q;

                    // half time step
                    float c = cs[2*it-2]; // please be in L1
                    float s = sn[2*it-2];
                    if (tid < ndof)
                    {
                        float dq = s_z[tid] + c * F;
                        dq += s * G;
                        dq *= inv_mi;
                        
                        s_p_half[tid] = p - half_dt * q;
                        s_q_half[tid] = q + half_dt * dq;

                        s_z[tid] = 0.0f;
                        p -= dt * s_q_half[tid]; // <- full time step
                    }

                    if (tid < fdof)
                        s_z[tid] -= Hi * s_q_half[tid];
                    __syncthreads();

                    stiffness(k, l, el, g_tid, s_I, s_D, s_q_half, s_p_half, s_z);
                    __syncthreads();

                    // full time step + WaveHoltz update
                    dK = wh_filter[it]; // please be in L1
                    c = cs[2*it-1];
                    s = sn[2*it-1];
                    if (tid < ndof)
                    {
                        float dq = s_z[tid] + c * F;
                        dq += s * G;
                        dq *= inv_mi;

                        q += dt * dq;

                        u += dK * p;
                        v += dK * q;
                    }
                } // time stepping
            } // WaveHoltz

            // rescale v and update global solution
            if (tid < ndof)
            {
                v *= rw;

                const float M = mi * g_inv_m(tid, subsp);
                const double m_u = M * u;
                const double m_v = M * v;
                
                atomicAdd(y+g_idx, m_u);
                atomicAdd(y+g_ndof+g_idx, m_v);
            }

            // update Lambdas
            if (tid < fdof)
            {
                int idx = B(tid, 1, subsp);
                if (idx >= 0)
                {
                    float S = 2.0 * ai * omega;
                    lambda_update[idx] = -lambda - S * v;
                    mu_update[idx]     = -mu     + S * u;
                }
            }
        });

        // update Lambdas
        copy(2 * n_lambda, d_update, d_lambda);
    } // for lambda iteration
}

DDH::DDH(double omega_, const double * h_a, const H1Space& fem, int nx, int ny)
    : g_ndof{fem.size()},
      g_elem{fem.mesh().n_elem()},
      n_basis{fem.basis().size()},
      omega{omega_}
{
    // Setup domain decomposition by determining which elements belong to which
    // subdomain. For the structured meshes generated by Mesh2D::uniform_rect
    // this is straight forward.

    if (n_basis != 4 && n_basis != 8)
        cuddh_error("DDH error: Only n_basis==4, and n_basis==8 supported.");

    const int elems_per_domain_x = DDH_BLOCK_SIZE / n_basis;

    if (nx % elems_per_domain_x != 0 || ny % elems_per_domain_x != 0)
        cuddh_error("Only nx x ny meshes with nx and ny multiples of 32 / n_basis allowed.");

    const int num_domains_x = nx / elems_per_domain_x;
    const int num_domains_y = ny / elems_per_domain_x;

    n_domains = num_domains_x * num_domains_y;

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

    // Setup WaveHoltz by determining the time step and precomputing the time
    // filter K(t) scaled by the quadrature weights (trapezoid rule), and also
    // the time dependency of the forcing (sines and cosines). 
    double T = (2 * M_PI) / omega;
    double h = fem.mesh().min_h();
    dt = 0.5 * 0.5 * h / (n_basis * n_basis);
    nt = std::ceil(T / dt);
    dt = T / nt;

    _wh_filter.resize(nt+1);
    auto wh_filter = reshape(_wh_filter.host_write(), nt+1);
    for (int k = 0; k <= nt; ++k)
        wh_filter[k] = dt * (omega / M_PI) * (std::cos(omega * k * dt) - 0.25);
    wh_filter[0] *= 0.5;
    wh_filter[nt] *= 0.5;

    _cs.resize(2*nt+1);
    _sn.resize(2*nt+1);
    auto cs = reshape(_cs.host_write(), 2*nt+1);
    auto sn = reshape(_sn.host_write(), 2*nt+1);
    for (int k = 0; k <= 2*nt; ++k)
    {
        double t = 0.5 * k * dt;
        cs[k] = -std::cos(omega * t);
        sn[k] =  std::sin(omega * t);
    }
    
    // determine max subspace dimensions
    mx_dof = 0;
    mx_fdof = 0;
    auto sizes = efem->sizes(MemorySpace::HOST);
    auto fsizes = efem->fsizes(MemorySpace::HOST);
    for (int subsp = 0; subsp < n_domains; ++subsp)
    {
        mx_dof = std::max(mx_dof, sizes(subsp));
        mx_fdof = std::max(mx_fdof, fsizes(subsp));
    }

    auto h_n_elems = efem->n_elems(MemorySpace::HOST);
    auto h_elems = efem->elements(MemorySpace::HOST);
    mx_elem_per_dom = 0;
    for (int subsp = 0; subsp < n_domains; ++subsp)
    {
        const int s_nel = h_n_elems(subsp);
        mx_elem_per_dom = std::max(mx_elem_per_dom, s_nel);
    }

    // Set up lambdas for the substructured problem and compute index
    // information relating lambdas to their duals.

    auto cmap = efem->connectivity_map(MemorySpace::HOST);
    const int n_shared = cmap.shape(1);
    n_lambda = 2 * n_shared;
    _g_lambda.resize(2 * n_lambda); // g_lambda = (lambda1, lambda2, mu1, mu2)
    _g_update.resize(2 * n_lambda);

    // g_lambda = (lambda0, lambda1, mu0, mu1) where lambda0 is the "interior"
    // trace for each subspace and lambda1 is the "external" trace. cmap is a
    // one-to-one map between lambda1 and lambda2 in the respective face spaces
    // of each subspace, so we assign the column index of cmap to lambda0, and
    // take the index of lambda1 to be that of lambda0 offset by n_shared.
    // Consequently, the indices of mu0 and mu1 are those of lambda0 and
    // lambda1, respectively, offset by n_lambda (=2*n_shared).

    _Bf.resize(2 * mx_fdof * n_domains);
    auto B = reshape(_Bf.host_write(), mx_fdof, 2, n_domains);
    std::fill(B.begin(), B.end(), -1);

    for (int k = 0; k < n_shared; ++k)
    {
        int subspace0 = cmap(0, k);
        int subspace1 = cmap(1, k);
        int face_index0 = cmap(2, k);
        int face_index1 = cmap(3, k);

        B(face_index0, 0, subspace0) = k;
        B(face_index0, 1, subspace0) = n_shared+k;
        B(face_index1, 0, subspace1) = n_shared+k;
        B(face_index1, 1, subspace1) = k;
    }

    // compute permutation of DOFs so that the face DOFs are first
    imat perm(mx_dof, n_domains);
    imat inv_perm(mx_dof, n_domains);
    auto P = efem->face_proj(MemorySpace::HOST);
    for (int subsp = 0; subsp < n_domains; ++subsp)
    {
        const int ndof = sizes(subsp);
        const int fdof = fsizes(subsp);

        for (int i=0; i < ndof; ++i)
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

        for (int i=0; i < ndof; ++i)
        {
            if (contains(pp, i))
                continue;

            perm(l, subsp) = i;
            ++l;
        }

        for (int i=0; i < ndof; ++i)
        {
            int j = perm(i, subsp);
            inv_perm(j, subsp) = i;
        }
    }

    auto g_inds = efem->global_indices(MemorySpace::HOST);
    auto s_inds = efem->subspace_indices(MemorySpace::HOST);

    _gI.resize(mx_dof * n_domains);
    auto gI = reshape(_gI.host_write(), mx_dof, n_domains);
    
    _sI.resize(n_basis * n_basis * mx_elem_per_dom * n_domains);
    auto sI = reshape(_sI.host_write(), n_basis, n_basis, mx_elem_per_dom, n_domains);

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

    // Set up subspace finite element operators
    const Mesh2D& mesh = fem.mesh();
    const Basis& basis = fem.basis();
    const QuadratureRule& q = basis.quadrature();

    host_device_dvec _w(n_basis);
    double * h_w = _w.host_write();
    for (int i = 0; i < n_basis; ++i)
        h_w[i] = q.w(i);
    const double * d_w = _w.device_read();
    
    _D.resize(n_basis * n_basis);
    dmat D_temp(n_basis, n_basis);
    basis.deriv(n_basis, q.x(), D_temp);
    float * h_D = _D.host_write();
    for (int i=0; i < n_basis*n_basis; ++i)
        h_D[i] = D_temp[i];

    auto& metrics = mesh.element_metrics(q);
    const double * d_J = metrics.jacobians(MemorySpace::DEVICE);
    
    _g_tensor.resize(3 * n_basis * n_basis * mx_elem_per_dom * n_domains);
    float * d_G = _g_tensor.device_write();
    auto d_n_elems = efem->n_elems(MemorySpace::DEVICE);
    auto d_elems = efem->elements(MemorySpace::DEVICE);
    init_geom_factors(n_domains, mx_elem_per_dom, g_elem, n_basis, d_n_elems, d_elems, d_w, d_J, d_G);

    auto f_inds = efem->face_indices(MemorySpace::HOST);

    _m.resize(mx_dof * n_domains);
    auto m = reshape(_m.host_write(), mx_dof, n_domains);
    
    _H.resize(mx_fdof * n_domains);
    auto H = reshape(_H.host_write(), mx_fdof, n_domains);

    _a.resize(mx_dof * n_domains);
    auto A = reshape(_a.host_write(), mx_dof, n_domains);

    _gmi.resize(mx_dof * n_domains);
    auto gmi = reshape(_gmi.host_write(), mx_dof, n_domains);

    auto faces = efem->faces(MemorySpace::HOST);
    auto n_faces = efem->n_faces(MemorySpace::HOST);

    const double * h_detJ = metrics.measures(MemorySpace::HOST);
    auto detJ = reshape(h_detJ, n_basis, n_basis, g_elem);

    dvec mi(g_ndof);
    auto fem_gi = fem.global_indices(MemorySpace::HOST);
    
    for (int el = 0; el < g_elem; ++el)
        for (int j = 0; j < n_basis; ++j)
            for (int i = 0; i < n_basis; ++i)
                mi[fem_gi(i, j, el)] += q.w(i) * q.w(j) * detJ(i, j, el);
                
    for (int i=0; i < g_ndof; ++i)
        mi[i] = 1.0 / mi[i];
    
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

        const int s_dof = sizes(subsp);
        for (int i = 0; i < s_dof; ++i)
        {
            A(i, subsp) = h_a[gI(i, subsp)];
            gmi(i, subsp) = mi[gI(i, subsp)];
        }
        
        const int s_nf = n_faces(subsp);
        for (int f = 0; f < s_nf; ++f)
        {
            const int g_f = faces(f, subsp);
            const Edge * edge = fem.mesh().edge(g_f);

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

void DDH::action(const double * x, double * y) const
{
    auto B = reshape(_Bf.device_read(), mx_fdof, 2, n_domains);

    auto gI = reshape(_gI.device_read(), mx_dof, n_domains);
    auto sI = reshape(_sI.device_read(), n_basis, n_basis, mx_elem_per_dom, n_domains);

    auto D = reshape(_D.device_read(), n_basis, n_basis);
    
    auto g = reshape(_g_tensor.device_read(), 3, n_basis * n_basis * mx_elem_per_dom, n_domains);
    
    auto a = reshape(_a.device_read(), mx_dof, n_domains);
    auto m = reshape(_m.device_read(), mx_dof, n_domains);
    auto H = reshape(_H.device_read(), mx_fdof, n_domains);
    auto g_inv_m = reshape(_gmi.device_read(), mx_dof, n_domains);
    
    auto wh_filter = reshape(_wh_filter.device_read(), nt+1);
    auto cs = reshape(_cs.device_read(), 2*nt+1);
    auto sn = reshape(_sn.device_read(), 2*nt+1);

    float * d_lambda = _g_lambda.device_write();
    float * d_update = _g_update.device_write();

    if (n_basis == 4)
        ddh_action<4, DDH_BLOCK_SIZE/4>(efem.get(), g_ndof, n_domains, n_lambda, nt, omega, dt, B, gI, sI, D, g, m, g_inv_m, a, H, wh_filter, cs, sn, x, y, d_lambda, d_update);
    else if (n_basis == 8)
        ddh_action<8, DDH_BLOCK_SIZE/8>(efem.get(), g_ndof, n_domains, n_lambda, nt, omega, dt, B, gI, sI, D, g, m, g_inv_m, a, H, wh_filter, cs, sn, x, y, d_lambda, d_update);
    else
        cuddh_error("DDH::action only supports n_basis == 4 or 8.");
}
