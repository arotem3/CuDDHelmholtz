#include <map>

#include "DD2D/DDTraceFunc2D.hpp"
#include "DDHKernelImpl.hpp"
#ifdef CUDDH_HAS_CUDSS
#include "SparseMatrix.hpp"
#endif

using namespace cuddh;

template <typename scalar_t, SubdomainSolver Solver>
struct MakeSolverData;

template <typename scalar_t>
struct MakeSolverData<scalar_t, SubdomainSolver::WaveHoltz>
{
    static DDSolverData<scalar_t, SubdomainSolver::WaveHoltz> make(double omega, const GridFunc2D<double> &a,
                                                                   const EnsembleSpace &efem, int wh_iters)
    {
        cuddh_verify(wh_iters != 0,
                     printf("DDH error: waveholtz_iterations must be positive or -1 for residual-based stopping.\n"));
        return {make_DDWaveHoltz_2d<scalar_t>(efem, scalar_t(omega), &a), wh_iters};
    }
};

template <typename scalar_t>
struct MakeSolverData<scalar_t, SubdomainSolver::MINRES>
{
    static DDSolverData<scalar_t, SubdomainSolver::MINRES> make(double omega, const GridFunc2D<double> &a,
                                                                const EnsembleSpace &efem, int /*wh_iters*/)
    {
        GridFunc2D<double> a2 = a.transform([] __device__(double x) -> double { return x * x; });
        return {DDMassMatrix<scalar_t>(efem, a2), DDFaceMassMatrix<scalar_t>(efem, a), scalar_t(omega)};
    }
};

#ifdef CUDDH_HAS_CUDSS
template <typename scalar_t>
struct MakeSolverData<scalar_t, SubdomainSolver::SparseDirect>
{
    static DDSolverData<scalar_t, SubdomainSolver::SparseDirect> make(double omega, const GridFunc2D<double> &a,
                                                                      const EnsembleSpace &efem, int /*wh_iters*/)
    {
        const scalar_t oms = scalar_t(omega * omega);
        const scalar_t om = scalar_t(omega);

        GridFunc2D<double> a2 = a.transform([] __device__(double x) -> double { return x * x; });

        DDStiffnessMatrix<scalar_t> S_dd(efem);
        DDMassMatrix<scalar_t> M_dd(efem, a2);
        DDFaceMassMatrix<scalar_t> H_dd(efem, a);

        const auto blk_sizes = efem.sizes(MemorySpace::HOST);
        BlockSparseMatrix<scalar_t, true> A_blocks(efem.size(), blk_sizes.data());
        efem.set_pattern(A_blocks);
        A_blocks.finalize_pattern();

        S_dd.assemble(std::complex<scalar_t>(1, 0), A_blocks);
        M_dd.assemble(std::complex<scalar_t>(-oms, 0), A_blocks);
        H_dd.assemble(std::complex<scalar_t>(0, -om), A_blocks);

        A_blocks.finalize_values();

        SparseBlockLU<scalar_t, true> lu(A_blocks);

        const int buf_size = efem.size() * 2 * efem.max_size();
        return {std::move(lu), thrust::device_vector<scalar_t>(buf_size), thrust::device_vector<scalar_t>(buf_size)};
    }
};

namespace cuddh::details
{
    template <typename scalar_t>
    void invoke_sd_kernel(int n_domains, int mx_ndof, int mx_fdof, int g_ndof, int n_lambda, const int *s_ndof,
                          const int *s_fdof, const int *gI, const scalar_t *punity, const LambdaDOFData<scalar_t> *B,
                          const double *fem_in, double *fem_out, const scalar_t *lambda_in, scalar_t *lambda_out,
                          SparseBlockLU<scalar_t, true> &lu, scalar_t *d_rhs, scalar_t *d_sol);

    extern template void invoke_sd_kernel<float>(int, int, int, int, int, const int *, const int *, const int *,
                                                 const float *, const LambdaDOFData<float> *, const double *, double *,
                                                 const float *, float *, SparseBlockLU<float, true> &, float *,
                                                 float *);
    extern template void invoke_sd_kernel<double>(int, int, int, int, int, const int *, const int *, const int *,
                                                  const double *, const LambdaDOFData<double> *, const double *,
                                                  double *, const double *, double *, SparseBlockLU<double, true> &,
                                                  double *, double *);
} // namespace cuddh::details
#endif

static constexpr __device__ int2 get_indices(int t, int2 dims)
{
    return {.x = t % dims.x, .y = t / dims.x};
}

template <typename scalar_t>
static int lambda_dofs(thrust::device_vector<LambdaDOFData<scalar_t>> &B, const EnsembleSpace &efem, double omega,
                       TensorWrapper<2, const double> a_face)
{
    struct SharedDof
    {
        int subspaces[2];
        int local_dof_indices[2];
        double integral;
    };

    const int n_domains = efem.size();
    const int mx_fdof = efem.max_fsize();
    const int n_basis = efem.h1_space().basis().size();

    const Mesh2D &mesh = efem.h1_space().mesh();
    auto shared_faces = efem.shared_faces(MemorySpace::HOST);
    auto faces = efem.faces(MemorySpace::HOST);
    auto fI = efem.face_indices(MemorySpace::HOST);
    auto w = efem.h1_space().basis().quadrature().w(MemorySpace::HOST);

    std::map<int, std::map<int, SharedDof>> shared_dofs;
    const int n_shared_faces = shared_faces.shape(1);
    for (int s = 0; s < n_shared_faces; ++s)
    {
        const int domain0 = shared_faces(0, s);
        const int domain1 = shared_faces(1, s);
        const int local_face_index0 = shared_faces(2, s);
        const int local_face_index1 = shared_faces(3, s);

        const int global_face0 = faces(local_face_index0, domain0);
        const int global_face1 = faces(local_face_index1, domain1);
        cuddh_verify(global_face0 == global_face1, printf("DDH error: shared face indices do not match up."));

        const double edge_measure = mesh.edge(global_face0).measure();
        const int pair_key = std::min(domain0, domain1) + n_domains * std::max(domain0, domain1);
        auto &dofs = shared_dofs[pair_key];

        for (int i = 0; i < n_basis; ++i)
        {
            const int local_dof0 = fI(i, local_face_index0, domain0);
            const int local_dof1 = fI(i, local_face_index1, domain1);
            const int lkey = (domain0 < domain1) ? local_dof0 : local_dof1;

            if (not dofs.contains(lkey))
            {
                SharedDof dof{};
                dof.subspaces[0] = domain0;
                dof.subspaces[1] = domain1;
                dof.local_dof_indices[0] = local_dof0;
                dof.local_dof_indices[1] = local_dof1;
                dof.integral = 0.0;
                dofs[lkey] = dof;
            }

            dofs.at(lkey).integral += w(i) * edge_measure * (a_face(local_dof0, domain0) + a_face(local_dof1, domain1));
        }
    }

    int n_shared = 0;
    for (const auto &[_, dofs] : shared_dofs)
        n_shared += dofs.size();

    thrust::host_vector<LambdaDOFData<scalar_t>> h_B(2 * mx_fdof * n_domains, LambdaDOFData<scalar_t>{});
    auto b = reshape(thrust::raw_pointer_cast(h_B.data()), 2, mx_fdof, n_domains);

    int n_lambda = 2 * n_shared;
    int k = 0;
    for (const auto &[_, dofs] : shared_dofs)
    {
        for (const auto &[__, dof] : dofs)
        {
            const scalar_t T = std::sqrt(omega * dof.integral);

            for (const int s : {0, 1})
            {
                const int subspace = dof.subspaces[s];
                const int face_index = dof.local_dof_indices[s];

                for (const int o : {0, 1})
                {
                    if (b(o, face_index, subspace).i < 0)
                    {
                        b(o, face_index, subspace) = LambdaDOFData<scalar_t>{
                            .i = (s == 0) ? k : n_shared + k, .j = (s == 0) ? n_shared + k : k, .trOp = T};
                        break;
                    }
                }
            }

            ++k;
        }
    }

    B = h_B;
    return n_lambda;
}

static HostDeviceArray<double> face_dof_values(const EnsembleSpace &efem, const GridFunc2D<double> &a)
{
    const int nb = efem.h1_space().basis().size();
    const int mx_n_faces = efem.max_n_faces();
    const int mx_fdof = efem.max_fsize();
    const int n_domains = efem.size();

    DDTraceFunc2D<double> tr_a = subdomain_trace(efem, a);

    auto d_n_faces = efem.n_faces(MemorySpace::DEVICE);
    auto d_face_inds = efem.face_indices(MemorySpace::DEVICE);
    auto d_tr_a = tr_a.read(MemorySpace::DEVICE);

    HostDeviceArray<double> out(mx_fdof * n_domains);
    auto a_face = reshape(out.device_write(), mx_fdof, n_domains);

    forall_1d(nb, mx_n_faces * n_domains, [=] __device__(int tid) mutable {
        const int i = threadIdx.x;
        const int f = tid % mx_n_faces;
        const int subsp = tid / mx_n_faces;

        if (f >= d_n_faces(subsp))
            return;

        const int l = d_face_inds(i, f, subsp);
        a_face(l, subsp) = d_tr_a(i, f, subsp);
    });

    return out;
}

template <typename scalar_t>
static thrust::device_vector<scalar_t> partition_of_unity(const EnsembleSpace &efem)
{
    MassMatrix M(efem.h1_space());
    DDMassMatrix<double> DDM(efem);

    auto d_m = M.to_device();
    auto d_ddm = DDM.to_device();

    const int n_domains = efem.size();
    const int mx_dof = efem.max_size();

    auto sizes = efem.sizes(MemorySpace::DEVICE);
    auto gI = efem.global_indices(MemorySpace::DEVICE);

    thrust::device_vector<scalar_t> P(mx_dof * n_domains, 0);
    auto p = reshape(thrust::raw_pointer_cast(P.data()), mx_dof, n_domains);

    forall(mx_dof * n_domains, [=] __device__(int tid) mutable {
        const auto [i, subsp] = get_indices(tid, {mx_dof, n_domains});

        if (i >= sizes(subsp))
            return;

        p(i, subsp) = d_ddm(i, subsp) / d_m[gI(i, subsp)];
    });

    return P;
}

static DDKernelConfig make_valid_config(DDKernelConfig config, int nb, int mx_elems)
{
    int mx_dof = nb * nb * mx_elems;

    if (config.block_size == DDKernelConfig::Default)
    {
        if (config.tdof <= 0)
        {
            if (mx_dof <= 256)
            {
                config.block_size = DDKernelConfig::t256;
                config.tdof = 1;
            }
            else if (mx_dof <= 512)
            {
                config.block_size = DDKernelConfig::t512;
                config.tdof = 1;
            }
            else if (mx_dof <= 1024)
            {
                config.block_size = DDKernelConfig::t1024;
                config.tdof = 1;
            }
            else
            {
                config.block_size = DDKernelConfig::t1024;
                config.tdof = (mx_dof + 1023) / 1024;
            }
        }
        else
        {
            int B = (mx_dof + config.tdof - 1) / config.tdof;
            cuddh_verify(
                B <= 1024,
                printf("DDH: Kernel configuration with tdof = %d requires %d threads/block which "
                       "exceeds the maximum of 1024. This occured because at least one subdomain has %d elements.\n",
                       config.tdof, B, mx_elems));

            if (B <= 256)
                config.block_size = DDKernelConfig::t256;
            else if (B <= 512)
                config.block_size = DDKernelConfig::t512;
            else
                config.block_size = DDKernelConfig::t1024;
        }
    }
    else
    {
        int B = static_cast<int>(config.block_size);
        int t = (mx_dof + B - 1) / B;

        if (config.tdof <= 0)
            config.tdof = t;
        cuddh_verify(config.tdof >= t,
                     printf("DDH: Kernel configuration with %d threads/block requires tdof >= %d, but tdof = %d "
                            "was specified. This occured because at least one subdomain has %d elements.\n",
                            B, t, config.tdof, mx_elems));
    }

    cuddh_verify(config.tdof <= 4, printf("DDH: Kernel configuration with tdof > 4 not compiled.\n"));
    return config;
}

template <typename scalar_t, SubdomainSolver Solver>
DDSubstructuredOperator<scalar_t, Solver>::DDSubstructuredOperator(const EnsembleSpace &efem, double omega,
                                                                   const GridFunc2D<double> &a, DDKernelConfig config,
                                                                   int waveholtz_iterations)
#ifdef CUDDH_HAS_CUDSS
    requires(Solver != SubdomainSolver::SparseDirect)
#endif
    : Operator<scalar_t>(0),
      DDSolverData<scalar_t, Solver>{MakeSolverData<scalar_t, Solver>::make(omega, a, efem, waveholtz_iterations)},
      g_ndof{efem.h1_space().size()},
      g_elem{efem.h1_space().mesh().n_elem()},
      n_basis{efem.h1_space().basis().size()},
      efem{efem},
      S(efem)
{
    n_domains = efem.size();
    mx_fdof = efem.max_fsize();
    mx_elem_per_dom = efem.max_n_elem();
    mx_dof = efem.max_size();

    kernel_config = make_valid_config(config, n_basis, mx_elem_per_dom);

    if (kernel_config.tdof > 1)
    {
        const int work_size = static_cast<int>(kernel_config.block_size) * kernel_config.tdof * n_domains;
        _work.resize(work_size);
    }

    _partition_of_unity = partition_of_unity<scalar_t>(efem);
    CUDDH_CUDA_CHECK(cudaDeviceSynchronize());

    HostDeviceArray<double> a_face = face_dof_values(efem, a);
    n_lambda = lambda_dofs(_B, efem, omega, reshape(a_face.read(MemorySpace::HOST), mx_fdof, n_domains));
    this->set_size(2 * n_lambda);
}

#ifdef CUDDH_HAS_CUDSS
template <typename scalar_t, SubdomainSolver Solver>
DDSubstructuredOperator<scalar_t, Solver>::DDSubstructuredOperator(const EnsembleSpace &efem, double omega,
                                                                   const GridFunc2D<double> &a)
    requires(Solver == SubdomainSolver::SparseDirect)
    : Operator<scalar_t>(0),
      DDSolverData<scalar_t, Solver>{MakeSolverData<scalar_t, Solver>::make(omega, a, efem, 0)},
      g_ndof{efem.h1_space().size()},
      g_elem{efem.h1_space().mesh().n_elem()},
      n_basis{efem.h1_space().basis().size()},
      efem{efem},
      S(efem)
{
    n_domains = efem.size();
    mx_fdof = efem.max_fsize();
    mx_elem_per_dom = efem.max_n_elem();
    mx_dof = efem.max_size();

    _partition_of_unity = partition_of_unity<scalar_t>(efem);
    CUDDH_CUDA_CHECK(cudaDeviceSynchronize());

    HostDeviceArray<double> a_face = face_dof_values(efem, a);
    n_lambda = lambda_dofs(_B, efem, omega, reshape(a_face.read(MemorySpace::HOST), mx_fdof, n_domains));
    this->set_size(2 * n_lambda);
}
#endif

template <typename scalar_t, SubdomainSolver Solver>
void DDSubstructuredOperator<scalar_t, Solver>::action(const double *fem_in, double *fem_out, const scalar_t *lambda_in,
                                                       scalar_t *lambda_out) const
{
    const LambdaDOFData<scalar_t> *B = thrust::raw_pointer_cast(_B.data());
    const scalar_t *punity = thrust::raw_pointer_cast(_partition_of_unity.data());
    scalar_t *d_work = thrust::raw_pointer_cast(_work.data());
    const int bs = static_cast<int>(kernel_config.block_size);

    if constexpr (Solver == SubdomainSolver::WaveHoltz)
    {
        details::invoke_wh_kernel<scalar_t>(n_basis, kernel_config.tdof, bs, efem, g_ndof, n_lambda, B, S, punity,
                                            this->W, this->waveholtz_iterations, fem_in, fem_out, lambda_in, lambda_out,
                                            d_work);
    }
    else if constexpr (Solver == SubdomainSolver::MINRES)
    {
        auto sm = this->mass.to_device();
        auto sfm = this->face_mass.to_device();
        details::invoke_mr_kernel<scalar_t>(n_basis, kernel_config.tdof, bs, efem, g_ndof, n_lambda, B, S, punity, sm,
                                            sfm, this->omega, fem_in, fem_out, lambda_in, lambda_out, d_work);
    }
#ifdef CUDDH_HAS_CUDSS
    else if constexpr (Solver == SubdomainSolver::SparseDirect)
    {
        scalar_t *d_rhs = thrust::raw_pointer_cast(this->d_rhs.data());
        scalar_t *d_sol = thrust::raw_pointer_cast(this->d_sol.data());
        details::invoke_sd_kernel<scalar_t>(
            n_domains, mx_dof, mx_fdof, g_ndof, n_lambda, efem.sizes(MemorySpace::DEVICE).data(),
            efem.fsizes(MemorySpace::DEVICE).data(), efem.global_indices(MemorySpace::DEVICE).data(), punity, B, fem_in,
            fem_out, lambda_in, lambda_out, this->lu, d_rhs, d_sol);
    }
#endif
}

template <typename scalar_t, SubdomainSolver Solver>
void DDSubstructuredOperator<scalar_t, Solver>::action(const scalar_t *x, scalar_t *y) const
{
    action((const double *)nullptr, (double *)nullptr, x, y);
    symmetrize_ddh(n_lambda, x, y);
}

template <typename scalar_t, SubdomainSolver Solver>
void DDSubstructuredOperator<scalar_t, Solver>::rhs(const double *f, scalar_t *b) const
{
    action(f, (double *)nullptr, (const scalar_t *)nullptr, b);
    symmetrize_ddh(n_lambda, (const scalar_t *)nullptr, b);
}

template <typename scalar_t, SubdomainSolver Solver>
void DDSubstructuredOperator<scalar_t, Solver>::postprocess(const scalar_t *lambda, const double *f, double *y) const
{
    action(f, y, lambda, (scalar_t *)nullptr);
}

namespace cuddh
{
    template class DDSubstructuredOperator<float>;
    template class DDSubstructuredOperator<double>;
    template class DDSubstructuredOperator<float, SubdomainSolver::MINRES>;
    template class DDSubstructuredOperator<double, SubdomainSolver::MINRES>;

    template class DDH<float>;
    template class DDH<double>;
    template class DDH<float, SubdomainSolver::MINRES>;
    template class DDH<double, SubdomainSolver::MINRES>;

#ifdef CUDDH_HAS_CUDSS
    template class DDSubstructuredOperator<float, SubdomainSolver::SparseDirect>;
    template class DDSubstructuredOperator<double, SubdomainSolver::SparseDirect>;

    template class DDH<float, SubdomainSolver::SparseDirect>;
    template class DDH<double, SubdomainSolver::SparseDirect>;
#endif
} // namespace cuddh
