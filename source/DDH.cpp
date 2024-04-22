#include "DDH.hpp"
#include <iostream>

using namespace cuddh;

static void stiffness(int subsp,
                      int n_elem,
                      int n_basis,
                      const TensorWrapper<4,const double>& G,
                      const TensorWrapper<4,const int>& I,
                      const const_imat_wrapper& elems,
                      const dmat& D,
                      const double * u,
                      double * out)
{
    dcube F(2, n_basis, n_basis);
    
    for (int el = 0; el < n_elem; ++el)
    {
        const int g_el = elems(el, subsp);

        // compute contravariant flux
        for (int l = 0; l < n_basis; ++l)
        {
            for (int k = 0; k < n_basis; ++k)
            {
                double Dx = 0.0;
                double Dy = 0.0;
                for (int i = 0; i < n_basis; ++i)
                {
                    Dx += D(k, i) * u[I(i, l, el, subsp)];
                    Dy += D(l, i) * u[I(k, i, el, subsp)];
                }

                const double A = G(0, k, l, g_el);
                const double B = G(1, k, l, g_el);
                const double C = G(2, k, l, g_el);

                F(0, k, l) = A * Dx + B * Dy;
                F(1, k, l) = B * Dx + C * Dy;
            }
        }

        // inner product with D'
        for (int l = 0; l < n_basis; ++l)
        {
            for (int k = 0; k < n_basis; ++k)
            {
                double Su = 0.0;
                for (int i = 0; i < n_basis; ++i)
                {
                    Su += D(i, k) * F(0, i, l) + D(i, l) * F(1, k, i);
                }

                const int idx = I(k, l, el, subsp);
                out[idx] += Su;
            }
        }
    }
}

DDH::DDH(double omega_, const H1Space& fem, int nx, int ny)
    : g_ndof{fem.size()},
      g_elem{fem.mesh().n_elem()},
      n_basis{fem.basis().size()},
      omega{omega_},
      g_inv_m(fem)
{
    n_domains = 2;
    lambda_maxit = 2;
    wh_maxit = 20;

    // waveholtz setup
    double T = (2 * M_PI) / omega;
    double h = fem.mesh().min_h();
    dt = 0.5 * h / (n_basis * n_basis);
    nt = std::ceil(T / dt);
    dt = T / nt;

    wh_filter.reshape(nt+1);
    for (int k = 0; k <= nt; ++k)
        wh_filter(k) = dt * (omega / M_PI) * (std::cos(omega * k * dt) - 0.25);
    wh_filter(0) *= 0.5;
    wh_filter(nt) *= 0.5;

    cs.reshape(2*nt+1);
    sn.reshape(2*nt+1);
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
            element_labels(i, j) = (2 * i < nx) ? 0 : 1;
        }
    }

    efem.reset(new EnsembleSpace(fem, n_domains, element_labels));

    auto cmap = efem->connectivity_map();
    int _n = cmap.shape(1);
    n_lambda = 2 * _n;
    g_lambda.reshape(2 * n_lambda); // g_lambda = (lambda1, lambda2, mu1, mu2)
    g_update.reshape(2 * n_lambda);

    // g_lambda = (lambda1, lambda2, mu1, mu2) where lambda1 is the "interior"
    // trace for each subspace and lambda2 is the "external" trace. cmap is a
    // one-to-one map between lambda1 and lambda2 in the respective face spaces
    // of each subspace, so we assign the column index of cmap to lambda1, and
    // take the index of lambda2 to be twice that of lambda1. Consequently, the
    // indices of mu1 and mu2 are those of lambda1 and lambda2 offset by n_lambda.

    std::unordered_map<int, std::vector<std::pair<int,int>>> _b;
    std::unordered_map<int, std::vector<std::pair<int,int>>> _bt;
    for (int k = 0; k < _n; ++k)
    {
        int subspace0 = cmap(0, k);
        int subspace1 = cmap(1, k);
        int face_index0 = cmap(2, k);
        int face_index1 = cmap(3, k);

        _b[subspace0].push_back({face_index0, k}); // lambda0
        _b[subspace1].push_back({face_index1, 2*k}); // lambda0 of subspace1 == lambda1 of subspace0

        _bt[subspace0].push_back({face_index0, 2*k}); // lambda1
        _bt[subspace1].push_back({face_index1, k}); // lambda1 of subpsace1 == lambda0 of subspace0
    }

    s_lambda.reshape(n_domains);
    int mx = 0;
    for (int p = 0; p < n_domains; ++p)
    {
        const int n = _b.at(p).size();
        s_lambda(p) = n;
        mx = std::max(mx, n);
    }

    B.reshape(2, mx, n_domains); B.fill(-1);
    dualB.reshape(2, mx, n_domains); dualB.fill(-1);

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
    
    D.reshape(n_basis, n_basis);
    basis.deriv(n_basis, q.x(), D);

    auto& metrics = fem.mesh().element_metrics(q);
    auto J = reshape(metrics.jacobians(), 2, 2, n_basis, n_basis, g_elem);
    
    auto g_inds = fem.global_indices();
    auto s_inds = efem->subspace_indices();
    auto f_inds = efem->face_indices();

    g_tensor.reshape(3, n_basis, n_basis, g_elem);
    for (int el = 0; el < g_elem; ++el)
    {
        for (int j = 0; j < n_basis; ++j)
        {
            for (int i = 0; i < n_basis; ++i)
            {
                const double W = q.w(i) * q.w(j);
                const double Y_eta = J(1, 1, i, j, el);
                const double X_eta = J(0, 1, i, j, el);
                const double Y_xi  = J(1, 0, i, j, el);
                const double X_xi  = J(0, 0, i, j, el);

                const double detJ = X_xi * Y_eta - X_eta * Y_xi;
                
                g_tensor(0, i, j, el) =  W * (Y_eta * Y_eta + X_eta * X_eta) / detJ;
                g_tensor(1, i, j, el) = -W * (Y_xi  * Y_eta + X_xi  * X_eta) / detJ;
                g_tensor(2, i, j, el) =  W * (Y_xi  * Y_xi  + X_xi  * X_xi)  / detJ;
            }
        }
    }

    int mx_dof = 0, mx_fdof = 0;
    auto sizes = efem->sizes();
    auto fsizes = efem->fsizes();
    for (int subsp = 0; subsp < n_domains; ++subsp)
    {
        mx_dof = std::max(mx_dof, sizes(subsp));
        mx_fdof = std::max(mx_fdof, fsizes(subsp));
    }

    inv_m.reshape(mx_dof, n_domains);
    m.reshape(mx_dof, n_domains);
    H.reshape(mx_fdof, n_domains);

    auto elems = efem->elements();
    auto n_elems = efem->n_elems();
    auto faces = efem->faces();
    auto n_faces = efem->n_faces();
    
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
                    const double detJ = J(0, 0, i, j, g_el) * J(1, 1, i, j, g_el) - J(0, 1, i, j, g_el) * J(1, 0, i, j, g_el);
                    m(l, subsp) += q.w(i) * q.w(j) * detJ;
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
    auto gI = efem->global_indices(); // global indices of subspace local dofs
    auto s_dof = efem->sizes();  // number of subdomain degrees of freedom
    auto s_fdof = efem->fsizes(); // number of face space degrees of freedom

    auto elems = efem->elements(); // elements for each subdomain
    auto faces = efem->faces(); // the lambda-faces for each subdomain

    auto s_elem = efem->n_elems(); // number of elements for each subdomain
    auto s_faces = efem->n_faces(); // number of boundary faces for each subdomain

    auto subspace_indices = efem->subspace_indices(); // ((i,j,el), p) -> p-th subspace index for dof (i,j,el)
    auto P = efem->face_proj(); // (i, p) -> p-th subspace's face space index to subspace index

    auto g = reshape(g_tensor.data(), 3, n_basis, n_basis, g_elem);
    const double half_dt = 0.5 * dt;
    const double rw = 1.0 / omega;

    g_lambda.zeros();

    const double * g_F = x;
    const double * g_G = x + g_ndof;
    double * U = y;
    double * V = y + g_ndof;
    
    for (int lit = 0; lit < lambda_maxit; ++lit)
    {
        for (int i = 0; i < g_ndof; ++i)
        {
            U[i] = 0.0;
            V[i] = 0.0;
        }

        std::cout << lit << " / " << lambda_maxit << "\n";
        for (int subsp = 0; subsp < n_domains; ++subsp)
        {
            // get subspace dimensions
            const int nel = s_elem(subsp);
            const int nf = s_faces(subsp);

            // copy global lambda to subdomain face space
            const int nl = s_lambda(subsp);
            const int fdof = s_fdof(subsp);
            const int ndof = s_dof(subsp);

            // note nl <= fdof. The lambdas make up a subset of the boundary
            // degrees of freedom. The rest is outflow corresponding to lambda = 0

            dvec lambda(fdof), mu(fdof);

            for (int i = 0; i < nl; ++i)
            {
                const int j = B(0, i, subsp); // face space index
                const int idx = B(1, i, subsp); // global lambda index
                lambda[j] = g_lambda[idx];
                mu[j] = g_lambda[idx + n_lambda];
            }

            // copy global x to forcing
            dvec F(ndof), G(ndof);
            for (int i = 0; i < ndof; ++i)
            {
                const int idx = gI(i, subsp);
                F[i] = g_F[idx];
                G[i] = g_G[idx];
            }

            // Add trace terms to forcing
            for (int i = 0; i < fdof; ++i)
            {
                const int idx = P(i, subsp);
                const double Hi = H(i, subsp);
                F[idx] += Hi * lambda[i];
                G[idx] += Hi * mu[i];
            }

            // WaveHoltz/time-stepping work variables
            dvec u(ndof), v(ndof),
                 p(ndof), q(ndof),
                 p_half(ndof), q_half(ndof),
                 z(ndof);

            // WaveHoltz iteration
            for (int whit=0; whit < wh_maxit; ++whit)
            {
                double dK = wh_filter(0);
                for (int i = 0; i < ndof; ++i)
                {
                    p[i] = u[i];
                    q[i] = v[i];

                    u[i] = dK * u[i];
                    v[i] = dK * v[i];
                }

                // time stepping loop
                for (int it=1; it <= nt; ++it)
                {
                    // compute dq/dt
                    for (int i = 0; i < ndof; ++i)
                        z[i] = 0.0;
                    
                    stiffness(subsp, nel, n_basis, g, subspace_indices, elems, D, p, z); // z <- S * p

                    // z <- z - H * q
                    for (int i = 0; i < fdof; ++i)
                    {
                        const int idx = P(i, subsp); // volume index of trace
                        const double Hi = H(i, subsp);
                        z[idx] -= Hi * q[idx];
                    }

                    // half time step
                    double c = cs(2*it-2);
                    double s = sn(2*it-2);
                    for (int i = 0; i < ndof; ++i)
                    {
                        const double dq = z[i] - c * F[i] + s * G[i];
                        p_half[i] = p[i] - half_dt * q[i];
                        q_half[i] = q[i] + half_dt * inv_m(i, subsp) * dq;
                    }

                    for (int i = 0; i < ndof; ++i)
                        z[i] = 0.0;
                    
                    stiffness(subsp, nel, n_basis, g, subspace_indices, elems, D, p_half, z);

                    for (int i = 0; i < fdof; ++i)
                    {
                        const int idx = P(i, subsp);
                        const double Hi = H(i, subsp);
                        z[idx] -= Hi * q_half[idx];
                    }

                    // full time step + WaveHoltz update
                    dK = wh_filter(it);
                    c = cs(2*it-1);
                    s = sn(2*it-1);
                    for (int i = 0; i < ndof; ++i)
                    {
                        const double dq = z[i] - c * F[i] + s * G[i];
                        p[i] -= dt * q_half[i];
                        q[i] += dt * inv_m(i, subsp) * dq;

                        u[i] += dK * p[i];
                        v[i] += dK * q[i];
                    }
                } // time stepping
                std::cout << "\t" << std::setw(5) << whit << " / " << wh_maxit << " : " << subsp << "\r" << std::flush;
            } // WaveHoltz
            std::cout << std::endl;

            // rescale v
            for (int i = 0; i < ndof; ++i)
                v[i] *= rw;

            // update Lambdas
            for (int i = 0; i < nl; ++i)
            {
                const int j = dualB(0, i, subsp);
                const int idx = dualB(1, i, subsp);
                const int k = P(j, subsp); // volume index of trace

                g_update[idx] = -lambda[j] - 2.0 * omega * v[k];
                g_update[idx + n_lambda] = -mu[j] + 2.0 * omega * u[k];
            }

            // add M * u to global solution
            for (int i=0; i < ndof; ++i)
            {
                const int idx = gI(i, subsp);
                const double mi = m(i, subsp);
                U[idx] += mi * u[i];
                V[idx] += mi * v[i];
            }
        } // for subspace

        // update Lambdas
        for (int i = 0; i < 2*n_lambda; ++i)
            g_lambda[i] = g_update[i];
    } // for lambda iteration

    // multiply by inverse mass
    g_inv_m.action(U, U);
    g_inv_m.action(V, V);
}

void DDH::action(double c, const double * x, double * y) const
{
    cuddh_error("DDH::action(c, x, y) is not implemented");
}