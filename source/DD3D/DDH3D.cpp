#include "DD3D/DDH3D.hpp"

#include "FixedTensorWrapper.hpp"

#define DD3D_MX_DOF 512

using namespace cuddh;

/* implements ddh_action where each thread is resposible for a single DOF within an element and `EL_PER_THR` elements
 * per thread. Taking `EL_PER_THR > 1` sacrifices some parallelism in exchange for larger subdomains.
 */
template <typename scalar_t, int NB, int NEL>
static void ddh_action_dof_per_thread(
    const EnsembleSpace3D &efem, const int g_ndof, /* global finite element degrees of freedom */
    const int n_lambda,                            /* number of substructured DOFs (lambda) */
    const TensorWrapper<3, const int2> B,          /* global lambda indices associated with boundary DOFs */
    const TensorWrapper<3, const scalar_t> T,      /* lambda trace operator */
    const DeviceDDStiffnessMatrix3D<scalar_t> stiffness_matrix,
    const MatrixWrapper<const scalar_t> punity,    /* partition of unity */
    const DeviceDDWaveHoltz3D<scalar_t> waveholtz, /* waveholtz */
    const double *const __restrict__ d_x,          /* input */
    double *const __restrict__ d_y,                /* output */
    const scalar_t *const __restrict__ d_lambda,   /* substructured problem input DOFs */
    scalar_t *const __restrict__ d_update          /* substructured problem output DOFs */
)
{
    using vec2 = cuddh::scalar2<scalar_t>;
    using vec3 = cuddh::scalar3<scalar_t>;
    using sym3x3 = SmallSymmetricMatrix<scalar_t, 3>;

    constexpr int wh_maxit = 20;

    constexpr int EDOF = NB * NB * NB;  // number of DOFs per element
    constexpr int MX_NDOF = EDOF * NEL; // maximum number of degrees of freedom per thread block

    const int n_domains = efem.size();

    auto gI = efem.global_indices(MemorySpace::DEVICE); // global solution DOF associated with subdomain DOFs

    auto subsp_elems = efem.n_elems(MemorySpace::DEVICE); // number of elements in a subdoamin
    auto subsp_sizes = efem.sizes(MemorySpace::DEVICE);   // number of subdomain degrees of freedom
    auto subsp_fsizes = efem.fsizes(MemorySpace::DEVICE); // number of face space degrees of freedom

    if (d_y)
        dla::zeros(2 * g_ndof, d_y);

    const scalar_t *g_lambda = (d_lambda) ? d_lambda : nullptr;
    const scalar_t *g_mu = (d_lambda) ? (d_lambda + n_lambda) : nullptr;

    scalar_t *lambda_update = (d_update) ? d_update : nullptr;
    scalar_t *mu_update = (d_update) ? (d_update + n_lambda) : nullptr;

    forall_2d(EDOF, NEL, n_domains, [=] __device__(const int subsp) mutable -> void {
        using BStiffness = SubdomainStiffnessMatrix<scalar_t, NB, NEL>;

        const int tid = threadIdx.x + EDOF * threadIdx.y; // linearized thread index.

        // get subspace dimensions
        const int fdof = subsp_fsizes(subsp); // number of face space degrees of freedom
        const int ndof = subsp_sizes(subsp);  // number of subdomain degrees of freedom

        cuddh_assert(ndof <= MX_NDOF,
                     printf("DDH3D error: exceeded maximum number of subdomain DOFs per thread block (%d > %d)\n", ndof,
                            MX_NDOF););
        cuddh_assert(subsp_elems(subsp) <= NEL,
                     printf("DDH3D error: subdomain larger than block thread dimension can accomodate.\n"));

        __shared__ typename BStiffness::SharedResources smem;
        auto A = stiffness_matrix.template subspace_op<NB, NEL>(subsp, subsp_elems(subsp), smem);
        auto evolve_project = waveholtz.subspace_op(subsp, tid, ndof);

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

        vec2 u{0, 0};
        for (int it = 0; it < wh_maxit; ++it)
        {
            u = evolve_project(A, u, F);
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
                       const EnsembleSpace3D &efem, double omega, VectorWrapper<const double> a)
{
    const int n_domains = efem.size();
    const int mx_fdof = efem.max_fsize();

    auto cmap = efem.connectivity_map(MemorySpace::HOST);
    auto gI = efem.global_indices(MemorySpace::HOST);
    const int n_shared = cmap.shape(0);
    const int n_lambda = 2 * n_shared;

    // leading dim 3: at most 3 boundary faces can share a DOF in 3D
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
                        static_cast<scalar_t>(2.0 * omega * a(gI(face_index, subspace)) * dof.face_mass);
                    break;
                }
            }
        }
    }

    return n_lambda;
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
DDSubstructedProblem3D<scalar_t>::DDSubstructedProblem3D(double omega_, const double *h_a, const H1Space3D &fem,
                                                         const EnsembleSpace3D &efem_)
    : g_ndof{fem.size()},
      g_elem{fem.mesh().n_elem()},
      n_basis{fem.basis().size()},
      efem{efem_},
      S(fem, efem_),
      W(omega_, h_a, fem, efem_)
{
    cuddh_verify(n_basis >= 2 && n_basis <= 4, printf("DDH3D error: Only n_basis in [2,3,4] supported.\n"););
    cuddh_verify(efem.max_size() <= DD3D_MX_DOF, printf("DDH3D error: subdomains too big.\n"));

    n_domains = efem.size();
    mx_dof = efem.max_size();
    mx_fdof = efem.max_fsize();
    mx_elem_per_dom = efem.max_n_elem();

    n_lambda = lambda_dofs(_B, _T, efem, omega_, reshape(h_a, fem.size()));

    _partition_of_unity = partition_of_unity<scalar_t>(fem, efem);
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
    auto punity = reshape(_partition_of_unity, mx_dof, n_domains);

    (*func)(efem, g_ndof, n_lambda, B, T, S.to_device(), punity, W.to_device(), fem_in, fem_out, lambda_in, lambda_out);
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
