#include "DDHKernelImpl.hpp"

using namespace cuddh;

template <typename scalar_t, SubdomainSolver Solver>
struct MakeSolverData;

template <typename scalar_t>
struct MakeSolverData<scalar_t, SubdomainSolver::WaveHoltz>
{
    static DDSolverData<scalar_t, SubdomainSolver::WaveHoltz> make(double omega, const double *h_a,
                                                                   const H1Space2D &fem, const EnsembleSpace &efem,
                                                                   int wh_iters)
    {
        cuddh_verify(wh_iters != 0,
                     printf("DDH error: waveholtz_iterations must be positive or -1 for residual-based stopping.\n"));
        return {make_DDWaveHoltz_2d<scalar_t>(scalar_t(omega), h_a, fem, efem), wh_iters};
    }
};

template <typename scalar_t>
struct MakeSolverData<scalar_t, SubdomainSolver::MINRES>
{
    static DDSolverData<scalar_t, SubdomainSolver::MINRES> make(double omega, const double *h_a, const H1Space2D &fem,
                                                                const EnsembleSpace &efem, int /*wh_iters*/)
    {
        const int mx_dof = efem.max_size();
        const int mx_fdof = efem.max_fsize();
        const int n_domains = efem.size();

        DDMassMatrix<scalar_t> raw_M(fem, efem);
        DDFaceMassMatrix<scalar_t> raw_H(fem, efem);

        auto m_raw = raw_M.to_device(); // shape [mx_dof,  n_domains]
        auto h_raw = raw_H.to_device(); // shape [mx_fdof, n_domains]
        auto gI = efem.global_indices(MemorySpace::DEVICE);
        auto sizes = efem.sizes(MemorySpace::DEVICE);
        auto fsizes = efem.fsizes(MemorySpace::DEVICE);

        HostDeviceArray<scalar_t> scaled_mass(mx_dof * n_domains);
        {
            auto sm = reshape(scaled_mass.device_write(), mx_dof, n_domains);
            const double *d_a = h_a;
            forall(mx_dof * n_domains, [=] __device__(int tid) mutable {
                int subsp = tid / mx_dof;
                int i = tid % mx_dof;
                if (i >= sizes(subsp))
                {
                    sm(i, subsp) = scalar_t(0);
                    return;
                }
                scalar_t ai = static_cast<scalar_t>(d_a[gI(i, subsp)]);
                sm(i, subsp) = ai * ai * m_raw(i, subsp);
            });
        }

        HostDeviceArray<scalar_t> scaled_face_mass(mx_fdof * n_domains);
        {
            auto sfm = reshape(scaled_face_mass.device_write(), mx_fdof, n_domains);
            const double *d_a = h_a;
            forall(mx_fdof * n_domains, [=] __device__(int tid) mutable {
                int subsp = tid / mx_fdof;
                int i = tid % mx_fdof;
                if (i >= fsizes(subsp))
                {
                    sfm(i, subsp) = scalar_t(0);
                    return;
                }
                scalar_t ai = static_cast<scalar_t>(d_a[gI(i, subsp)]);
                sfm(i, subsp) = ai * h_raw(i, subsp);
            });
        }

        return {std::move(scaled_mass), std::move(scaled_face_mass), scalar_t(omega)};
    }
};

static constexpr __device__ int2 get_indices(int t, int2 dims)
{
    return {.x = t % dims.x, .y = t / dims.x};
}

template <typename scalar_t>
static int lambda_dofs(thrust::device_vector<LambdaDOFData<scalar_t>> &B, const EnsembleSpace &efem, double omega,
                       VectorWrapper<const double> a)
{
    const int n_domains = efem.size();
    const int mx_fdof = efem.max_fsize();

    auto cmap = efem.connectivity_map(MemorySpace::HOST);
    auto gI = efem.global_indices(MemorySpace::HOST);
    const int n_shared = cmap.shape(0);

    thrust::host_vector<LambdaDOFData<scalar_t>> h_B(2 * mx_fdof * n_domains, LambdaDOFData<scalar_t>{});
    auto b = reshape(thrust::raw_pointer_cast(h_B.data()), 2, mx_fdof, n_domains);

    int n_lambda = 2 * n_shared;
    for (int k = 0; k < n_shared; ++k)
    {
        const auto &dof = cmap(k);

        for (const int s : {0, 1})
        {
            const int subspace = dof.subspaces[s];
            const int face_index = dof.local_dof_indices[s];

            for (const int o : {0, 1})
            {
                if (b(o, face_index, subspace).i < 0)
                {
                    const scalar_t T = std::sqrt(2.0 * omega * a(gI(face_index, subspace)) * dof.face_mass);

                    b(o, face_index, subspace) = LambdaDOFData<scalar_t>{
                        .i = (s == 0) ? k : n_shared + k, .j = (s == 0) ? n_shared + k : k, .trOp = T};
                    break;
                }
            }
        }
    }

    B = h_B;
    return n_lambda;
}

template <typename scalar_t>
static thrust::device_vector<scalar_t> partition_of_unity(const H1Space2D &fem, const EnsembleSpace &efem)
{
    MassMatrix M(fem);
    DDMassMatrix<double> DDM(fem, efem);

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
DDSubstructuredOperator<scalar_t, Solver>::DDSubstructuredOperator(double omega, const double *h_a,
                                                                   const H1Space2D &fem, const EnsembleSpace &efem,
                                                                   DDKernelConfig config, int waveholtz_iterations)
    : DDSolverData<scalar_t, Solver>{
          MakeSolverData<scalar_t, Solver>::make(omega, h_a, fem, efem, waveholtz_iterations)},
      g_ndof{fem.size()},
      g_elem{fem.mesh().n_elem()},
      n_basis{fem.basis().size()},
      efem{efem},
      S(fem, efem)
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

    _partition_of_unity = partition_of_unity<scalar_t>(fem, efem);
    CUDDH_CUDA_CHECK(cudaDeviceSynchronize());
    n_lambda = lambda_dofs(_B, efem, omega, reshape(h_a, fem.size()));
}

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
    else
    {
        auto sm = reshape(this->scaled_mass.device_read(), mx_dof, n_domains);
        auto sfm = reshape(this->scaled_face_mass.device_read(), mx_fdof, n_domains);
        details::invoke_mr_kernel<scalar_t>(n_basis, kernel_config.tdof, bs, efem, g_ndof, n_lambda, B, S, punity, sm,
                                            sfm, this->omega, fem_in, fem_out, lambda_in, lambda_out, d_work);
    }
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
} // namespace cuddh
