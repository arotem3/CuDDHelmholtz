#include "DD3D/DDWaveHoltz3D.hpp"

#include "DD3D/DDFaceMassMatrix3D.hpp"
#include "DD3D/DDMassMatrix3D.hpp"

using namespace cuddh;

template <typename scalar_t>
static HostDeviceArray<cuddh::scalar2<scalar_t>> make_alpha_beta(const EnsembleSpace3D &efem, double theta,
                                                                 double sigma, const DDMassMatrix3D &M,
                                                                 const DDFaceMassMatrix3D &H)
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

    forall(mx_dof * n_domains, [=] __device__(int tid) mutable -> void {
        const int i = tid % mx_dof;
        const int subsp = tid / mx_dof;

        if (i >= s_dof(subsp))
            return;

        cuddh::scalar2<scalar_t> ab_i{0, 0};

        scalar_t Mi = m(i, subsp);
        scalar_t Hi = (i < s_fdof(subsp)) ? h(i, subsp) : scalar_t(0);

        const scalar_t inv = 1 / (Mi + theta * Hi);
        ab_i.x = (Mi - theta * Hi) * inv;
        ab_i.y = sigma * inv;

        alpha_beta(i, subsp) = ab_i;
    });

    return ab;
}

static double compute_dt(double h, double p, const GridFunc3D<double> *a)
{
    double reciprocal_max_vel = 1.0;
    if (a)
    {
        auto cinv = a->read(MemorySpace::DEVICE);
        auto begin = thrust::device_pointer_cast(cinv.data());
        reciprocal_max_vel = *thrust::min_element(begin, begin + cinv.size());
    }

    cuddh_verify(
        reciprocal_max_vel > 0,
        printf("DDWaveHoltz3D error: Coefficient must be positive. Encountered value: %f.\n", reciprocal_max_vel));

    return 2.0 * reciprocal_max_vel * h / (p * p);
}

template <typename scalar_t>
DDWaveHoltz<scalar_t> cuddh::make_DDWaveHoltz_3d(const EnsembleSpace3D &efem, scalar_t omega,
                                                 const GridFunc3D<double> *a)
{
    const H1Space3D &fem = efem.h1_space();

    DDWaveHoltz<scalar_t> W;
    W.n_domains = efem.size();
    W.mx_ndof = efem.max_size();
    W.omega = omega;

    double dt = compute_dt(fem.mesh().h(), fem.basis().size(), a);

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

    std::unique_ptr<GridFunc3D<double>> a2;
    if (a)
    {
        a2 = std::make_unique<GridFunc3D<double>>(a->transform([] __device__(double x) -> double { return x * x; }));
    }

    auto M = (a) ? DDMassMatrix3D(efem, *a2) : DDMassMatrix3D(efem);
    auto H = (a) ? DDFaceMassMatrix3D(efem, *a) : DDFaceMassMatrix3D(efem);
    W.alpha_beta = make_alpha_beta<scalar_t>(efem, theta, W.sigma, M, H);

    return W;
}

namespace cuddh
{
    template DDWaveHoltz<float> make_DDWaveHoltz_3d<float>(const EnsembleSpace3D &efem, float omega,
                                                           const GridFunc3D<double> *a);
    template DDWaveHoltz<double> make_DDWaveHoltz_3d<double>(const EnsembleSpace3D &efem, double omega,
                                                             const GridFunc3D<double> *a);
} // namespace cuddh
