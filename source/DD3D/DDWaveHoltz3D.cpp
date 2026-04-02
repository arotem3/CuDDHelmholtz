#include "DD3D/DDWaveHoltz3D.hpp"

using namespace cuddh;

template <typename scalar_t>
static HostDeviceArray<cuddh::scalar2<scalar_t>> make_alpha_beta(double theta, double sigma,
                                                                 const VectorWrapper<const double> a,
                                                                 const H1Space3D &fem, const EnsembleSpace3D &efem)
{
    DDMassMatrix3D M(fem, efem);
    DDFaceMassMatrix3D H(fem, efem);

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

        scalar_t ai = a(gI(i, subsp));
        scalar_t Mi = m(i, subsp);
        scalar_t Hi = (i < s_fdof(subsp)) ? h(i, subsp) : scalar_t(0);

        Hi *= ai;
        Mi *= ai * ai;

        const scalar_t inv = 1 / (Mi + theta * Hi);
        ab_i.x = (Mi - theta * Hi) * inv;
        ab_i.y = sigma * inv;

        alpha_beta(i, subsp) = ab_i;
    });

    return ab;
}

template <typename scalar_t>
DDWaveHoltz<scalar_t> cuddh::make_DDWaveHoltz_3d(scalar_t omega, const double *a, const H1Space3D &fem,
                                                 const EnsembleSpace3D &efem)
{
    DDWaveHoltz<scalar_t> W;
    W.n_domains = efem.size();
    W.mx_ndof = efem.max_size();
    W.omega = omega;

    double dt = [&]() {
        const int n_basis = fem.basis().size();
        const double h = fem.mesh().h();
        auto begin = thrust::device_pointer_cast(a);
        const double reciprocal_max_vel = *thrust::min_element(begin, begin + fem.size());
        return dt = 2.0 * reciprocal_max_vel * h / (n_basis * n_basis);
    }();

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

    W.alpha_beta = make_alpha_beta<scalar_t>(theta, W.sigma, reshape(a, fem.size()), fem, efem);

    return W;
}

namespace cuddh
{
    template DDWaveHoltz<float> make_DDWaveHoltz_3d<float>(float omega, const double *a, const H1Space3D &fem,
                                                           const EnsembleSpace3D &efem);
    template DDWaveHoltz<double> make_DDWaveHoltz_3d<double>(double omega, const double *a, const H1Space3D &fem,
                                                             const EnsembleSpace3D &efem);
} // namespace cuddh
