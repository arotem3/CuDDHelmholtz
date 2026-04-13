#include "DD2D/DDWaveHoltz2D.hpp"

using namespace cuddh;

static constexpr __device__ int2 get_indices(int t, int2 dims)
{
    return {.x = t % dims.x, .y = t / dims.x};
}

template <typename scalar_t>
static HostDeviceArray<cuddh::scalar2<scalar_t>> make_alpha_beta(const EnsembleSpace &efem, double theta, double sigma,
                                                                 const DDMassMatrix<scalar_t> &M,
                                                                 const DDFaceMassMatrix<scalar_t> &H)
{
    auto m = M.to_device();
    auto h = H.to_device();

    const int n_domains = efem.size();
    const int mx_dof = efem.max_size();

    auto s_dof = efem.sizes(MemorySpace::DEVICE);
    auto s_fdof = efem.fsizes(MemorySpace::DEVICE);
    auto gI = efem.global_indices(MemorySpace::DEVICE);

    HostDeviceArray<cuddh::scalar2<scalar_t>> ab(mx_dof * n_domains);
    auto alpha_beta = reshape(ab.device_write(), mx_dof, n_domains);

    forall(mx_dof * n_domains, [=] __device__(int tid) mutable {
        const auto [i, subsp] = get_indices(tid, {mx_dof, n_domains});

        const int ndof = s_dof(subsp);
        const int fdof = s_fdof(subsp);

        if (i >= ndof)
            return;

        scalar_t Mi = m(i, subsp);
        scalar_t Hi = (i < fdof) ? h(i, subsp) : 0;

        scalar_t inv = 1 / (Mi + theta * Hi);
        scalar_t alpha = (Mi - theta * Hi) * inv;
        scalar_t beta = sigma * inv;

        alpha_beta(i, subsp) = {alpha, beta};
    });

    return ab;
}

static double compute_dt(double h, double p, const GridFunc2D<double> *a)
{
    double reciprocal_max_vel = 1.0;
    if (a)
    {
        auto cinv = a->read(MemorySpace::DEVICE);
        auto begin = thrust::device_pointer_cast(cinv.data());
        reciprocal_max_vel = *thrust::min_element(begin, begin + cinv.size());
    }

    return 2.0 * reciprocal_max_vel * h / (p * p);
}

template <typename scalar_t>
DDWaveHoltz<scalar_t> cuddh::make_DDWaveHoltz_2d(const EnsembleSpace &efem, scalar_t omega, const GridFunc2D<double> *a)
{
    DDWaveHoltz<scalar_t> W;
    W.n_domains = efem.size();
    W.mx_ndof = efem.max_size();
    W.omega = omega;

    double dt = compute_dt(efem.h1_space().mesh().h(), efem.h1_space().basis().size(), a);

    const double T = 2 * M_PI / omega;
    W.nt = std::ceil(T / dt);
    dt = T / W.nt;

    W.C = std::cos(0.5 * omega * dt);
    W.S = std::sin(0.5 * omega * dt);

    const double tn = std::tan(0.5 * omega * dt);
    const double a0 = 0.25 * (1 - tn * tn);

    W.weight = 2.0 / W.nt;
    W.shift = W.weight * a0;

    const double theta = tn / omega;
    W.sigma = W.S / (0.5 * omega);

    std::unique_ptr<GridFunc2D<double>> a2;
    if (a)
        a2 = std::make_unique<GridFunc2D<double>>(a->transform([] __device__(double x) -> double { return x * x; }));

    auto M = (a) ? DDMassMatrix<scalar_t>(efem, *a2) : DDMassMatrix<scalar_t>(efem);
    auto H = (a) ? DDFaceMassMatrix<scalar_t>(efem, *a) : DDFaceMassMatrix<scalar_t>(efem);
    W.alpha_beta = make_alpha_beta<scalar_t>(efem, theta, W.sigma, M, H);

    return W;
}

namespace cuddh
{
    template DDWaveHoltz<float> make_DDWaveHoltz_2d<float>(const EnsembleSpace &efem, float omega,
                                                           const GridFunc2D<double> *a);
    template DDWaveHoltz<double> make_DDWaveHoltz_2d<double>(const EnsembleSpace &efem, double omega,
                                                             const GridFunc2D<double> *a);
} // namespace cuddh
