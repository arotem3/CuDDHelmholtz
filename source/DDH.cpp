#include "DDH.hpp"
#include <iostream>

// one dimensional size of each domain decomp block. Each block has
// DDH_BLOCK_SIZE * DDH_BLOCK_SIZE degrees of freedom.
#define DDH_BLOCK_SIZE 16

using namespace cuddh;

static void init_geom_factors(int n_elem,
                              int n_basis,
                              const double * d_w,
                              const double * d_J,
                              float * d_G)
{
    auto w = reshape(d_w, n_basis);
    auto J = reshape(d_J, 2, 2, n_basis, n_basis, n_elem);

    auto G = reshape(d_G, 3, n_basis, n_basis, n_elem);

    forall_2d(n_basis, n_basis, n_elem, [=] __device__ (int el) mutable -> void
    {
        const int i = threadIdx.x;
        const int j = threadIdx.y;

        const double W = w(i) * w(j);
        const double Y_eta = J(1, 1, i, j, el);
        const double X_eta = J(0, 1, i, j, el);
        const double Y_xi  = J(1, 0, i, j, el);
        const double X_xi  = J(0, 0, i, j, el);

        const double detJ = X_xi * Y_eta - X_eta * Y_xi;
        
        G(0, i, j, el) =  W * (Y_eta * Y_eta + X_eta * X_eta) / detJ;
        G(1, i, j, el) = -W * (Y_xi  * Y_eta + X_xi  * X_eta) / detJ;
        G(2, i, j, el) =  W * (Y_xi  * Y_xi  + X_xi  * X_xi)  / detJ;
    });
}

template <int NB>
static void __device__ stiffness(int subsp,
                                int n_elem,
                                int n_basis,
                                const TensorWrapper<4,const float>& G,
                                const TensorWrapper<4,const int>& I,
                                const const_imat_wrapper& elems,
                                const float D[NB][NB],
                                float * w, /* <- work */
                                float * u, /* input and work */
                                float * out)
{
    // get element and 2D tensor product indices from linear thread index
    const int el = threadIdx.x / (n_basis * n_basis);
    const int l = (threadIdx.x % (n_basis * n_basis)) / n_basis;
    const int k = threadIdx.x % n_basis;

    const int g_el = elems(el, subsp); // please be in L1

    // compute contravariant flux
    float Dx = 0.0, Dy = 0.0;
    for (int i = 0; i < n_basis; ++i)
    {
        Dx += D[k][i] * u[I(i, l, el, subsp)];
        Dy += D[l][i] * u[I(k, i, el, subsp)];
    }

    __syncthreads(); // <- we overwrite u in a second so this is critical

    const float A = G(0, k, l, g_el); // please be in L1
    const float B = G(1, k, l, g_el);
    const float C = G(2, k, l, g_el);

    auto F0 = reshape(u, n_basis, n_basis, n_elem);
    auto F1 = reshape(w, n_basis, n_basis, n_elem);

    F0(k, l, el) = A * Dx + B * Dy;
    F1(k, l, el) = B * Dx + C * Dy;

    __syncthreads();

    // inner product with D'
    float Su = 0.0;
    for (int i = 0; i < n_basis; ++i)
        Su += D[i][k] * F0(i, l, el) + D[i][l] * F1(k, i, el);
    
    const int idx = I(k, l, el, subsp);
    atomicAdd(out + idx, Su);
}

template <int NB, int NEL>
static void ddh_action(const EnsembleSpace * efem,
                       int g_ndof,
                       int g_elem,
                       int n_domains,
                       int n_basis,
                       int n_lambda,
                       int nt,
                       double omega,
                       double dt,
                       const DiagInvMassMatrix& g_inv_m,
                       const_ivec_wrapper s_lambda,
                       const_icube_wrapper B,
                       const_icube_wrapper dualB,
                       MatrixWrapper<const float> _D,
                       TensorWrapper<4, const float> g,
                       MatrixWrapper<const float> m,
                       MatrixWrapper<const float> a,
                       MatrixWrapper<const float> H,
                       VectorWrapper<const float> wh_filter,
                       VectorWrapper<const float> cs,
                       VectorWrapper<const float> sn,
                       const double * __restrict__ x,
                       double * __restrict__ y,
                       float * __restrict__ d_lambda,
                       float * __restrict__ d_update)
{
    constexpr int lambda_maxit = 40;
    constexpr int wh_maxit = 10;

    auto gI = efem->global_indices(MemorySpace::DEVICE); // global indices of subspace local dofs
    auto s_dof = efem->sizes(MemorySpace::DEVICE);  // number of subdomain degrees of freedom
    auto s_fdof = efem->fsizes(MemorySpace::DEVICE); // number of face space degrees of freedom

    auto elems = efem->elements(MemorySpace::DEVICE); // elements for each subdomain
    auto faces = efem->faces(MemorySpace::DEVICE); // the lambda-faces for each subdomain

    auto s_elem = efem->n_elems(MemorySpace::DEVICE); // number of elements for each subdomain
    
    auto subspace_indices = efem->subspace_indices(MemorySpace::DEVICE); // ((i,j,el), p) -> p-th subspace index for dof (i,j,el)
    auto P = efem->face_proj(MemorySpace::DEVICE); // (i, p) -> p-th subspace's face space index to i-th subspace DOF

    const float half_dt = 0.5 * dt;
    const float rw = 1.0 / omega;

    auto g_lambda = reshape(d_lambda, n_lambda);
    auto g_mu = reshape(d_lambda + n_lambda, n_lambda);

    auto lambda_update = reshape(d_update, n_lambda);
    auto mu_update = reshape(d_update + n_lambda, n_lambda);

    auto g_F = reshape(x, g_ndof);
    auto g_G = reshape(x+g_ndof, g_ndof);
    auto U = reshape(y, g_ndof);
    auto V = reshape(y+g_ndof, g_ndof);

    zeros(2 * n_lambda, d_lambda);

    constexpr int MX_NDOF = NB * NB * NEL * NEL; // == DDH_BLOCK_SIZE^2

    for (int lit = 0; lit < lambda_maxit; ++lit)
    {
        zeros(2 * g_ndof, y);

        forall_1d(MX_NDOF, n_domains, [=] __device__ (int subsp) mutable -> void
        {
            const int tid = threadIdx.x; // thread id

            // get subspace dimensions
            const int nel = s_elem(subsp); // number of elements in subspace
            const int nl = s_lambda(subsp); // number of lambdas in facespace
            const int fdof = s_fdof(subsp); // dimension of facespace
            const int ndof = s_dof(subsp); // dimension of subspace

            // note nl <= fdof. The lambdas make up a subset of the boundary
            // degrees of freedom. The rest is outflow corresponding to lambda = 0

            // shared mem
            __shared__ float F[MX_NDOF];
            __shared__ float G[MX_NDOF];
            __shared__ float u[MX_NDOF];
            __shared__ float v[MX_NDOF];
            __shared__ float p[MX_NDOF];
            __shared__ float q[MX_NDOF];
            __shared__ float p_half[MX_NDOF];
            __shared__ float q_half[MX_NDOF];
            __shared__ float z[MX_NDOF];
            __shared__ float D[NB][NB];

            // copy D
            if (tid < NB * NB)
            {
                int i = tid % NB;
                int j = tid / NB;
                D[i][j] = _D(i, j);
            }

            // each thread has an associated lambda DOF, subspace DOF, and
            // FaceSpace DOF. The following variables are indices of those DOFs
            // in various arrays. Additionally, the elements of the (diagonal)
            // mass matrices are also kept.

            int g_idx; // subspace DOF[tid] global index. gI(tid, subsp)
            int pi; // facespace DOF[tid] subspace index. P(tid, subsp)

            float Hi; // H(tid, subsp)
            float Mi; // inv_m(tid, subsp)
            float lambda; // lambda[tid]
            float mu; // mu[tid]

            if (tid < fdof)
            {
                pi = P(tid, subsp);
                Hi = a(pi, subsp) * H(tid, subsp);
            }

            // copy global x to forcing, init work variables
            if (tid < ndof)
            {
                g_idx = gI(tid, subsp);
                F[tid] = g_F[g_idx];
                G[tid] = g_G[g_idx];

                u[tid] = 0.0f;
                v[tid] = 0.0f;

                const float ai = a(tid, subsp);
                Mi = ai * ai * m(tid, subsp);
                Mi = 1.0f / Mi;
            }
            __syncthreads();

            // add lambda to forcing, init work variables
            if (tid < nl)
            {
                const int j = B(0, tid, subsp); // face space index
                const int idx = B(1, tid, subsp); // global lambda index
                const int k = P(j, subsp); // subspace index

                lambda = g_lambda[idx];
                mu = g_mu[idx];

                float Hl = H(j, subsp);

                F[k] += Hl * lambda;
                G[k] += Hl * mu;
            }

            // WaveHoltz iteration
            // p, q are the time dependant variables.
            // u, v are the solution estimates.
            for (int whit=0; whit < wh_maxit; ++whit)
            {
                __syncthreads();
                
                float dK = wh_filter(0); // please be in L1
                if (tid < ndof)
                {
                    p[tid] = u[tid];
                    q[tid] = v[tid];

                    u[tid] = dK * u[tid];
                    v[tid] = dK * v[tid];
                }

                // time stepping
                for (int it=1; it <= nt; ++it)
                {
                    // to save shared memory we use p_half and q_half as work
                    // variables in the computation of the stiffness action. So
                    // we copy p to p_half and pass p_half to stifness which
                    // will compute the action on p_half but also overwrite it
                    // with junk.

                    if (tid < ndof)
                    {
                        z[tid] = 0.0f;
                        p_half[tid] = p[tid];
                    }
                    __syncthreads();

                    // z <- z + S * p, overwrites p and uses q_half as a work variable
                    stiffness(subsp, nel, n_basis, g, subspace_indices, elems, D, q_half, p_half, z);
                    __syncthreads();

                    // z <- z - H * q
                    if (tid < fdof)
                        z[pi] -= Hi * q[pi];
                    __syncthreads();

                    // half time step
                    float c = cs(2*it-2); // please be in L1
                    float s = sn(2*it-2);
                    if (tid < ndof)
                    {
                        const float dq = z[tid] - c * F[tid] + s * G[tid];
                        
                        p_half[tid] = p[tid] - half_dt * q[tid];
                        q_half[tid] = q[tid] + half_dt * Mi * dq;

                        z[tid] = 0.0f;
                        p[tid] -= dt * q_half[tid]; // <- full time step
                    }
                    __syncthreads();

                    if (tid < fdof)
                        z[pi] -= Hi * q_half[pi];
                    __syncthreads();

                    stiffness(subsp, nel, n_basis, g, subspace_indices, elems, D, q_half, p_half, z);
                    __syncthreads();

                    // full time step + WaveHoltz update
                    dK = wh_filter(it); // please be in L1
                    c = cs(2*it-1);
                    s = sn(2*it-1);
                    if (tid < ndof)
                    {
                        const float dq = z[tid] - c * F[tid] + s * G[tid];
                        q[tid] += dt * Mi * dq;

                        u[tid] += dK * p[tid];
                        v[tid] += dK * q[tid];
                    }
                } // time stepping
            } // WaveHoltz
            __syncthreads();

            // rescale v and update global solution
            if (tid < ndof)
            {
                v[tid] *= rw;

                const double M  = m(tid, subsp);
                const double m_u = M * u[tid];
                const double m_v = M * v[tid];
                
                atomicAdd(U+g_idx, m_u);
                atomicAdd(V+g_idx, m_v);
            }
            __syncthreads();

            // update Lambdas
            if (tid < nl)
            {
                const int j = dualB(0, tid, subsp);
                const int idx = dualB(1, tid, subsp);
                const int k = P(j, subsp); // volume index of trace
                const double af = a(k, subsp);

                lambda_update[idx] = -lambda - 2.0f * af * omega * v[k];
                mu_update[idx]     = -mu     + 2.0f * af * omega * u[k];
            }
        });

        // update Lambdas
        copy(2 * n_lambda, d_update, d_lambda);

        // multiply by inverse mass
        g_inv_m.action(U, U);
        g_inv_m.action(V, V);
    } // for lambda iteration
}

DDH::DDH(double omega_, const double * h_a, const H1Space& fem, int nx, int ny)
    : g_ndof{fem.size()},
      g_elem{fem.mesh().n_elem()},
      n_basis{fem.basis().size()},
      omega{omega_},
      g_inv_m(fem)
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
        cs[k] = std::cos(omega * t);
        sn[k] = std::sin(omega * t);
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

    std::vector<std::vector<std::pair<int, int>>> _b(n_domains);
    std::vector<std::vector<std::pair<int, int>>> _bt(n_domains);
    for (int k = 0; k < n_shared; ++k)
    {
        int subspace0 = cmap(0, k);
        int subspace1 = cmap(1, k);
        int face_index0 = cmap(2, k);
        int face_index1 = cmap(3, k);

        _b.at(subspace0).push_back({face_index0, k}); // lambda0
        _b.at(subspace1).push_back({face_index1, n_shared + k}); // lambda0 of subspace1 == lambda1 of subspace0

        _bt.at(subspace0).push_back({face_index0, n_shared + k}); // lambda1
        _bt.at(subspace1).push_back({face_index1, k}); // lambda1 of subpsace1 == lambda0 of subspace0
    }

    _s_lambda.resize(n_domains);
    auto s_lambda = reshape(_s_lambda.host_write(), n_domains);
    mx_n_lambda = 0;
    for (int p = 0; p < n_domains; ++p)
    {
        const int n = _b.at(p).size();
        s_lambda(p) = n;
        mx_n_lambda = std::max(mx_n_lambda, n);
    }

    _B.resize(2 * mx_n_lambda * n_domains);
    auto B = reshape(_B.host_write(), 2, mx_n_lambda, n_domains);
    std::fill(B.begin(), B.end(), -1);
    
    _dualB.resize(2 * mx_n_lambda * n_domains);
    auto dualB = reshape(_dualB.host_write(), 2, mx_n_lambda, n_domains);
    std::fill(dualB.begin(), dualB.end(), -1);

    for (int p = 0; p < n_domains; ++p)
    {
        const int n = s_lambda(p);
        auto& s_b = _b.at(p);
        auto& s_bt = _bt.at(p);

        for (int l = 0; l < n; ++l)
        {
            auto [face_index0, lambda_index0] = s_b.at(l);
            B(0, l, p) = face_index0;
            B(1, l, p) = lambda_index0;

            auto [face_index1, lambda_index1] = s_bt.at(l);
            dualB(0, l, p) = face_index1;
            dualB(1, l, p) = lambda_index1;
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
    
    _g_tensor.resize(3 * n_basis * n_basis * g_elem);
    float * d_G = _g_tensor.device_write();
    init_geom_factors(g_elem, n_basis, d_w, d_J, d_G);

    auto g_inds = efem->global_indices(MemorySpace::HOST);
    auto s_inds = efem->subspace_indices(MemorySpace::HOST);
    auto f_inds = efem->face_indices(MemorySpace::HOST);

    mx_dof = 0;
    mx_fdof = 0;
    auto sizes = efem->sizes(MemorySpace::HOST);
    auto fsizes = efem->fsizes(MemorySpace::HOST);
    for (int subsp = 0; subsp < n_domains; ++subsp)
    {
        mx_dof = std::max(mx_dof, sizes(subsp));
        mx_fdof = std::max(mx_fdof, fsizes(subsp));
    }

    _m.resize(mx_dof * n_domains);
    auto m = reshape(_m.host_write(), mx_dof, n_domains);
    
    _H.resize(mx_fdof * n_domains);
    auto H = reshape(_H.host_write(), mx_fdof, n_domains);

    _a.resize(mx_dof * n_domains);
    auto A = reshape(_a.host_write(), mx_dof, n_domains);

    auto elems = efem->elements(MemorySpace::HOST);
    auto n_elems = efem->n_elems(MemorySpace::HOST);
    auto faces = efem->faces(MemorySpace::HOST);
    auto n_faces = efem->n_faces(MemorySpace::HOST);

    mx_elem_per_dom = 0;
    for (int subsp = 0; subsp < n_domains; ++subsp)
    {
        const int s_nel = n_elems(subsp);
        mx_elem_per_dom = std::max(mx_elem_per_dom, s_nel);
    }

    const double * h_detJ = metrics.measures(MemorySpace::HOST);
    auto detJ = reshape(h_detJ, n_basis, n_basis, g_elem);
    
    for (int subsp = 0; subsp < n_domains; ++subsp)
    {
        const int s_nel = n_elems(subsp);
        for (int el = 0; el < s_nel; ++el)
        {
            const int g_el = elems(el, subsp);
            for (int j = 0; j < n_basis; ++j)
            {
                for (int i = 0; i < n_basis; ++i)
                {
                    const int l = s_inds(i, j, el, subsp);
                    m(l, subsp) += q.w(i) * q.w(j) * detJ(i, j, g_el);
                }
            }
        }

        const int s_dof = sizes(subsp);
        for (int i = 0; i < s_dof; ++i)
            A(i, subsp) = h_a[g_inds(i, subsp)];
        
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
    auto B = reshape(_B.device_read(), 2, mx_n_lambda, n_domains);
    auto dualB = reshape(_dualB.device_read(), 2, mx_n_lambda, n_domains);
    auto s_lambda = reshape(_s_lambda.device_read(), n_domains);

    auto D = reshape(_D.device_read(), n_basis, n_basis);
    auto g = reshape(_g_tensor.device_read(), 3, n_basis, n_basis, g_elem);
    auto a = reshape(_a.device_read(), mx_dof, n_domains);
    auto m = reshape(_m.device_read(), mx_dof, n_domains);
    auto H = reshape(_H.device_read(), mx_fdof, n_domains);

    auto wh_filter = reshape(_wh_filter.device_read(), nt+1);
    auto cs = reshape(_cs.device_read(), 2*nt+1);
    auto sn = reshape(_sn.device_read(), 2*nt+1);

    float * d_lambda = _g_lambda.device_write();
    float * d_update = _g_update.device_write();

    if (n_basis == 4)
        ddh_action<4, DDH_BLOCK_SIZE/4>(efem.get(), g_ndof, g_elem, n_domains, n_basis, n_lambda, nt, omega, dt, g_inv_m, s_lambda, B, dualB, D, g, m, a, H, wh_filter, cs, sn, x, y, d_lambda, d_update);
    else if (n_basis == 8)
        ddh_action<8, DDH_BLOCK_SIZE/8>(efem.get(), g_ndof, g_elem, n_domains, n_basis, n_lambda, nt, omega, dt, g_inv_m, s_lambda, B, dualB, D, g, m, a, H, wh_filter, cs, sn, x, y, d_lambda, d_update);
    else
        cuddh_error("DDH::action only supports n_basis == 4 or 8.");
}
