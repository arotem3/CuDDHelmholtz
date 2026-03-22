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

    forall_1d(mx_dof, n_domains, [=] __device__(int subsp) mutable -> void {
        const int i = threadIdx.x;
        const int ndof = s_dof(subsp);
        const int fdof = s_fdof(subsp);

        cuddh::scalar2<scalar_t> ab_i{0, 0};

        if (i < ndof)
        {
            scalar_t ai = a(gI(i, subsp));
            scalar_t Mi = m(i, subsp);
            scalar_t Hi = (i < fdof) ? h(i, subsp) : scalar_t(0);

            Hi *= ai;
            Mi *= ai * ai;

            const scalar_t inv = 1 / (Mi + theta * Hi);
            ab_i.x = (Mi - theta * Hi) * inv;
            ab_i.y = sigma * inv;
        }

        alpha_beta(i, subsp) = ab_i;
    });

    CUDDH_CUDA_CHECK(cudaDeviceSynchronize());

    return ab;
}

template <typename scalar_t>
cuddh::DDWaveHoltz3D<scalar_t>::DDWaveHoltz3D(scalar_t omega, const double *d_a, const H1Space3D &fem,
                                              const EnsembleSpace3D &efem)
    : n_domains{efem.size()}, mx_ndof{efem.max_size()}, omega{omega}
{
    double dt = [&]() {
        const int n_basis = fem.basis().size();
        const double h = fem.mesh().h();
        auto begin = thrust::device_pointer_cast(d_a);
        const double reciprocal_max_vel = *std::min_element(begin, begin + fem.size());
        return dt = 2.0 * reciprocal_max_vel * h / (n_basis * n_basis);
    }();

    const double T = 2 * M_PI / omega;
    nt = std::ceil(T / dt);
    dt = T / nt;

    C = std::cos(0.5 * omega * dt);
    S = std::sin(0.5 * omega * dt);

    const double tn = std::tan(0.5 * omega * dt);
    const double a0 = 0.25 * (1 - tn * tn);

    weight = 2.0 / nt;
    shift = weight * a0;

    const double theta = tn / omega;
    sigma = S / (0.5 * omega);

    alpha_beta = make_alpha_beta<scalar_t>(theta, sigma, reshape(d_a, fem.size()), fem, efem);
}

namespace cuddh
{
    template class DDWaveHoltz3D<float>;
    template class DDWaveHoltz3D<double>;
} // namespace cuddh
