#include "DDH.hpp"
#include <iostream>

using namespace cuddh;

static void init_geom_factors(int n_elem,
                              int n_basis,
                              const double * d_w,
                              const double * d_J,
                              double * d_G)
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
                                const TensorWrapper<4,const double>& G,
                                const TensorWrapper<4,const int>& I,
                                const const_imat_wrapper& elems,
                                const const_dmat_wrapper& D,
                                double F[][NB][NB][2],
                                const double * u,
                                double * out)
{
    const int k = threadIdx.x;
    const int l = threadIdx.y;
    const int el = threadIdx.z;

    const int g_el = elems(el, subsp);

    // compute contravariant flux
    double Dx = 0.0, Dy = 0.0;
    for (int i = 0; i < n_basis; ++i)
    {
        const int il = I(i, l, el, subsp);
        const int ki = I(k, i, el, subsp);
        Dx += D(k, i) * u[il];
        Dy += D(l, i) * u[ki];
    }

    const double A = G(0, k, l, g_el);
    const double B = G(1, k, l, g_el);
    const double C = G(2, k, l, g_el);

    F[el][k][l][0] = A * Dx + B * Dy;
    F[el][k][l][1] = B * Dx + C * Dy;

    __syncthreads();

    // inner product with D'
    double Su = 0.0;
    for (int i = 0; i < n_basis; ++i)
        Su += D(i, k) * F[el][i][l][0] + D(i, l) * F[el][k][i][1];
    
    const int idx = I(k, l, el, subsp);
    atomicAdd(out+idx, Su);
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
                       const_dmat_wrapper D,
                       TensorWrapper<4, const double> g,
                       const_dmat_wrapper m,
                       const_dmat_wrapper inv_m,
                       const_dmat_wrapper H,
                       const_dvec_wrapper wh_filter,
                       const_dvec_wrapper cs,
                       const_dvec_wrapper sn,
                       const double * __restrict__ x,
                       double * __restrict__ y,
                       double * __restrict__ d_lambda,
                       double * __restrict__ d_update)
{
    constexpr int lambda_maxit = 2;
    constexpr int wh_maxit = 5;

    auto gI = efem->global_indices(MemorySpace::DEVICE); // global indices of subspace local dofs
    auto s_dof = efem->sizes(MemorySpace::DEVICE);  // number of subdomain degrees of freedom
    auto s_fdof = efem->fsizes(MemorySpace::DEVICE); // number of face space degrees of freedom

    auto elems = efem->elements(MemorySpace::DEVICE); // elements for each subdomain
    auto faces = efem->faces(MemorySpace::DEVICE); // the lambda-faces for each subdomain

    auto s_elem = efem->n_elems(MemorySpace::DEVICE); // number of elements for each subdomain
    
    auto subspace_indices = efem->subspace_indices(MemorySpace::DEVICE); // ((i,j,el), p) -> p-th subspace index for dof (i,j,el)
    auto P = efem->face_proj(MemorySpace::DEVICE); // (i, p) -> p-th subspace's face space index to subspace index

    const double half_dt = 0.5 * dt;
    const double rw = 1.0 / omega;

    auto g_lambda = reshape(d_lambda, 2 * n_lambda);
    auto g_update = reshape(d_update, 2 * n_lambda);

    auto g_F = reshape(x, g_ndof);
    auto g_G = reshape(x+g_ndof, g_ndof);
    auto U = reshape(y, g_ndof);
    auto V = reshape(y+g_ndof, g_ndof);

    zeros(2 * n_lambda, d_lambda);

    constexpr int MX_NDOF = NB * NB * NEL * NEL;
    constexpr int MX_FDOF = 4 * (NB * NEL - NEL) - 4;

    for (int lit = 0; lit < lambda_maxit; ++lit)
    {
        zeros(2 * g_ndof, y);

        forall_3d(NB, NB, NEL*NEL, n_domains, [=] __device__ (int subsp) mutable -> void
        {
            // thread id
            const int t = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z); // linear thread index
            const int inc = blockDim.x * blockDim.y * blockDim.z;

            // get subspace dimensions
            const int nel = s_elem(subsp);

            const int nl = s_lambda(subsp);
            const int fdof = s_fdof(subsp);
            const int ndof = s_dof(subsp);

            // note nl <= fdof. The lambdas make up a subset of the boundary
            // degrees of freedom. The rest is outflow corresponding to lambda = 0

            // shared mem
            __shared__ double lambda[MX_FDOF];
            __shared__ double mu[MX_FDOF];
            __shared__ double F[MX_NDOF];
            __shared__ double G[MX_NDOF];
            __shared__ double u[MX_NDOF];
            __shared__ double v[MX_NDOF];
            __shared__ double p[MX_NDOF];
            __shared__ double q[MX_NDOF];
            __shared__ double p_half[MX_NDOF];
            __shared__ double q_half[MX_NDOF];
            __shared__ double z[MX_NDOF];
            __shared__ double flx[NEL*NEL][NB][NB][2];

            // copy global lambda to subdomain face space
            for (int i = t; i < nl; i += inc)
            {
                int j = B(0, i, subsp); // face space index
                int idx = B(1, i, subsp); // global lambda index
                
                lambda[j] = g_lambda[idx];
                mu[j] = g_lambda[idx + n_lambda];
            }

            // copy global x to forcing
            for (int i = t; i < ndof; i += inc)
            {
                const int idx = gI(i, subsp);
                F[i] = g_F[idx];
                G[i] = g_G[idx];
            }

            // zero out work variables
            for (int i = t; i < ndof; i += inc)
            {
                u[i] = 0.0;
                v[i] = 0.0;
            }

            __syncthreads();

            // Add trace terms to forcing
            for (int i = t; i < fdof; i += inc)
            {
                const int idx = P(i, subsp);
                const double Hi = H(i, subsp);
                F[idx] += Hi * lambda[i];
                G[idx] += Hi * mu[i];
            }

            // WaveHoltz iteration
            for (int whit=0; whit < wh_maxit; ++whit)
            {
                __syncthreads();
                
                double dK = wh_filter(0);
                for (int i = t; i < ndof; i += inc)
                {
                    p[i] = u[i];
                    q[i] = v[i];

                    u[i] = dK * u[i];
                    v[i] = dK * v[i];
                }

                // time stepping
                for (int it=1; it <= nt; ++it)
                {
                    for (int i = t; i < ndof; i += inc)
                        z[i] = 0.0;
                    __syncthreads();

                    stiffness(subsp, nel, n_basis, g, subspace_indices, elems, D, flx, p, z);
                    __syncthreads();

                    // z <- z - H * q
                    for (int i = t; i < fdof; i += inc)
                    {
                        const int idx = P(i, subsp);
                        const double Hi = H(i, subsp);

                        z[idx] -= Hi * q[idx];
                    }
                    __syncthreads();

                    // half time step
                    double c = cs(2*it-2);
                    double s = sn(2*it-2);
                    for (int i = t; i < ndof; i += inc)
                    {
                        const double dq = z[i] - c * F[i] + s * G[i];
                        p_half[i] = p[i] - half_dt * q[i];
                        q_half[i] = q[i] + half_dt * inv_m(i, subsp) * dq;
                    }
                    __syncthreads();

                    for (int i = t; i < fdof; i += inc)
                        z[i] = 0.0;

                    __syncthreads();

                    stiffness(subsp, nel, n_basis, g, subspace_indices, elems, D, flx, p_half, z);
                    __syncthreads();

                    for (int i = t; i < fdof; i += inc)
                    {
                        const int idx = P(i, subsp);
                        const double Hi = H(i, subsp);
                        z[idx] -= Hi * q_half[idx];
                    }

                    __syncthreads();

                    // full time step + WaveHoltz update
                    dK = wh_filter(it);
                    c = cs(2*it-1);
                    s = sn(2*it-1);
                    for (int i = t; i < ndof; i += inc)
                    {
                        const double dq = z[i] - c * F[i] + s * G[i];
                        p[i] -= dt * q_half[i];
                        q[i] += dt * inv_m(i, subsp) * dq;

                        u[i] += dK * p[i];
                        v[i] += dK * q[i];
                    }
                } // time stepping
            } // WaveHoltz

            __syncthreads();

            // rescale v and update global solution
            for (int i = t; i < ndof; i += inc)
            {
                v[i] *= rw;

                const int idx = gI(i, subsp);
                const double mi = m(i, subsp);
                const double m_u = mi * u[i];
                const double m_v = mi * v[i];
                
                atomicAdd(U+idx, m_u);
                atomicAdd(V+idx, m_v);
            }
                
            __syncthreads();

            // update Lambdas
            for (int i = t; i < nl; i += inc)
            {
                const int j = dualB(0, i, subsp);
                const int idx = dualB(1, i, subsp);
                const int k = P(j, subsp); // volume index of trace

                g_update[idx] = -lambda[j] - 2.0 * omega * v[k];
                g_update[idx + n_lambda] = -mu[j] + 2.0 * omega * u[k];
            }
        });

        // update Lambdas
        copy(2 * n_lambda, g_update, g_lambda);
    } // for lambda iteration

    // multiply by inverse mass
    g_inv_m.action(U, U);
    g_inv_m.action(V, V);
}

DDH::DDH(double omega_, const H1Space& fem, int nx, int ny)
    : g_ndof{fem.size()},
      g_elem{fem.mesh().n_elem()},
      n_basis{fem.basis().size()},
      omega{omega_},
      g_inv_m(fem)
{
    int block_size_x = 32;

    if (n_basis != 4 && n_basis != 8)
        cuddh_error("Only n_basis==4, and n_basis==8 supported.");

    int elems_per_domain_x = block_size_x / n_basis;

    if (nx % elems_per_domain_x != 0 || ny % elems_per_domain_x != 0)
        cuddh_error("Only nx x ny meshes with nx and ny multiples of 32 / n_basis allowed.");

    int num_domains_x = nx / elems_per_domain_x;
    int num_domains_y = ny / elems_per_domain_x;

    n_domains = num_domains_x * num_domains_y;

    // waveholtz setup
    double T = (2 * M_PI) / omega;
    double h = fem.mesh().min_h();
    dt = 0.5 * h / (n_basis * n_basis);
    nt = std::ceil(T / dt);
    dt = T / nt;

    _wh_filter.resize(nt+1);
    auto wh_filter = reshape(_wh_filter.host_write(), nt+1);
    for (int k = 0; k <= nt; ++k)
        wh_filter(k) = dt * (omega / M_PI) * (std::cos(omega * k * dt) - 0.25);
    wh_filter(0) *= 0.5;
    wh_filter(nt) *= 0.5;

    _cs.resize(2*nt+1);
    _sn.resize(2*nt+1);
    auto cs = reshape(_cs.host_write(), 2*nt+1);
    auto sn = reshape(_sn.host_write(), 2*nt+1);
    for (int k = 0; k <= 2*nt; ++k)
    {
        double t = 0.5 * k * dt;
        cs(k) = std::cos(omega * t);
        sn(k) = std::sin(omega * t);
    }

    // set up domain decomp
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

    auto cmap = efem->connectivity_map(MemorySpace::HOST);
    int _n = cmap.shape(1);
    n_lambda = 2 * _n;
    _g_lambda.resize(2 * n_lambda); // g_lambda = (lambda1, lambda2, mu1, mu2)
    _g_update.resize(2 * n_lambda);

    // g_lambda = (lambda0, lambda1, mu0, mu1) where lambda0 is the "interior"
    // trace for each subspace and lambda1 is the "external" trace. cmap is a
    // one-to-one map between lambda1 and lambda2 in the respective face spaces
    // of each subspace, so we assign the column index of cmap to lambda0, and
    // take the index of lambda1 to be that of lambda0 offset by _n.
    // Consequently, the indices of mu0 and mu1 are those of lambda0 and
    // lambda1, respectively, offset by n_lambda (=2*_n).

    std::unordered_map<int, std::vector<std::pair<int,int>>> _b;
    std::unordered_map<int, std::vector<std::pair<int,int>>> _bt;
    for (int k = 0; k < _n; ++k)
    {
        int subspace0 = cmap(0, k);
        int subspace1 = cmap(1, k);
        int face_index0 = cmap(2, k);
        int face_index1 = cmap(3, k);

        _b[subspace0].push_back({face_index0, k}); // lambda0
        _b[subspace1].push_back({face_index1, _n + k}); // lambda0 of subspace1 == lambda1 of subspace0

        _bt[subspace0].push_back({face_index0, _n + k}); // lambda1
        _bt[subspace1].push_back({face_index1, k}); // lambda1 of subpsace1 == lambda0 of subspace0
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

    // set up FEM
    auto& basis = fem.basis();
    auto& q = basis.quadrature();

    host_device_dvec _w(n_basis);
    double * h_w = _w.host_write();
    for (int i = 0; i < n_basis; ++i)
        h_w[i] = q.w(i);
    const double * d_w = _w.device_read();
    
    _D.resize(n_basis * n_basis);
    basis.deriv(n_basis, q.x(), _D.host_write());

    auto& metrics = fem.mesh().element_metrics(q);
    const double * d_J = metrics.jacobians(MemorySpace::DEVICE);
    
    _g_tensor.resize(3 * n_basis * n_basis * g_elem);
    double * d_G = _g_tensor.device_write();
    init_geom_factors(g_elem, n_basis, d_w, d_J, d_G);

    auto g_inds = fem.global_indices(MemorySpace::HOST);
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

    _inv_m.resize(mx_dof * n_domains);
    auto inv_m = reshape(_inv_m.host_write(), mx_dof, n_domains);
    
    _m.resize(mx_dof * n_domains);
    auto m = reshape(_m.host_write(), mx_dof, n_domains);
    
    _H.resize(mx_fdof * n_domains);
    auto H = reshape(_H.host_write(), mx_fdof, n_domains);

    auto elems = efem->elements(MemorySpace::HOST);
    auto n_elems = efem->n_elems(MemorySpace::HOST);
    auto faces = efem->faces(MemorySpace::HOST);
    auto n_faces = efem->n_faces(MemorySpace::HOST);

    const double * h_detJ = metrics.measures(MemorySpace::HOST);
    auto detJ = reshape(h_detJ, n_basis, n_basis, g_elem);
    
    mx_elem_per_dom = 0;
    for (int subsp = 0; subsp < n_domains; ++subsp)
    {
        const int s_nel = n_elems(subsp);
        mx_elem_per_dom = std::max(mx_elem_per_dom, s_nel);
        for (int el = 0; el < s_nel; ++el)
        {
            const int g_el = elems(el, subsp);
            for (int j = 0; j < n_basis; ++j)
            {
                for (int i = 0; i < n_basis; ++i)
                {
                    const int l = s_inds(i, j, el, subsp);
                    const double dX = detJ(i, j, g_el);
                    m(l, subsp) += q.w(i) * q.w(j) * dX;
                }
            }
        }

        const int s_dof = sizes(subsp);
        for (int i = 0; i < s_dof; ++i)
            inv_m(i, subsp) = 1.0 / m(i, subsp);
        
        const int s_nf = n_faces(subsp);
        for (int f = 0; f < s_nf; ++f)
        {
            const int g_f = faces(f, subsp);
            const Edge * edge = fem.mesh().edge(g_f);

            for (int i = 0; i < n_basis; ++i)
            {
                const double xi = q.x(i);
                const double detJ = edge->measure(xi);

                const int l = f_inds(i, f, subsp);
                H(l, subsp) += detJ * q.w(i);
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
    auto inv_m = reshape(_inv_m.device_read(), mx_dof, n_domains);
    auto m = reshape(_m.device_read(), mx_dof, n_domains);
    auto H = reshape(_H.device_read(), mx_fdof, n_domains);

    auto wh_filter = reshape(_wh_filter.device_read(), nt+1);
    auto cs = reshape(_cs.device_read(), 2*nt+1);
    auto sn = reshape(_sn.device_read(), 2*nt+1);

    double * d_lambda = _g_lambda.device_write();
    double * d_update = _g_update.device_write();

    if (n_basis == 4)
        ddh_action<4, 8>(efem.get(), g_ndof, g_elem, n_domains, n_basis, n_lambda, nt, omega, dt, g_inv_m, s_lambda, B, dualB, D, g, m, inv_m, H, wh_filter, cs, sn, x, y, d_lambda, d_update);
    else if (n_basis == 8)
        ddh_action<8, 4>(efem.get(), g_ndof, g_elem, n_domains, n_basis, n_lambda, nt, omega, dt, g_inv_m, s_lambda, B, dualB, D, g, m, inv_m, H, wh_filter, cs, sn, x, y, d_lambda, d_update);
    else
        cuddh_error("DDH::action only supports n_basis == 4 or 8.");
}

void DDH::action(double c, const double * x, double * y) const
{
    cuddh_error("DDH::action(c, x, y) is not implemented");
}