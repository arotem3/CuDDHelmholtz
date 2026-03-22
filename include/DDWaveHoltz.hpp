#pragma once

#include "DD2D/EnsembleSpace.hpp"
#include "DD3D/EnsembleSpace3D.hpp"
#include "cxmult.hpp"

namespace cuddh
{
    template <typename scalar_t>
    struct DeviceDDWaveHoltz;

    template <typename scalar_t>
    struct SubdomainWaveHoltz
    {
        const DeviceDDWaveHoltz<scalar_t> &wh;
        scalar_t alpha, beta;

        template <typename SubdomainStiffnessMatrix>
        __device__ scalar2<scalar_t> operator()(const SubdomainStiffnessMatrix &A, scalar2<scalar_t> u,
                                                const scalar2<scalar_t> &F) const
        {
            scalar_t cs = 1, sn = 0;
            scalar_t K = wh.filter(cs);

            scalar_t p = u.x;

            details::cxmult(cs, sn, wh.C, wh.S);

            scalar_t q = (-sn * u.x + cs * u.y) * wh.omega;

            u.x = K * p;

            K = wh.filter(cs);
            u.y = K * q;

            for (int n = 1; n < wh.nt; ++n)
            {
                details::cxmult(cs, sn, wh.C, wh.S);
                K = wh.filter(cs);

                p += wh.sigma * q;
                u.x += K * p;

                q = alpha * q + beta * (-A(p) + cs * F.x + sn * F.y);

                details::cxmult(cs, sn, wh.C, wh.S);
                K = wh.filter(cs);

                u.y += K * q;
            }

            u.y /= wh.omega;

            return u;
        }
    };

    template <typename scalar_t>
    struct DeviceDDWaveHoltz
    {
        int nt;
        scalar_t omega;

        scalar_t weight;
        scalar_t shift;

        scalar_t sigma;

        scalar_t C, S; // cos(omega * dt / 2), sin(omega * dt / 2)

        MatrixWrapper<const scalar2<scalar_t>> alpha_beta;

        inline constexpr scalar_t filter(scalar_t cs) const { return weight * cs - shift; }

        __device__ SubdomainWaveHoltz<scalar_t> subspace_op(int subsp, int tid, int ndof) const
        {
            const scalar2<scalar_t> ab = (tid < ndof) ? alpha_beta(tid, subsp) : scalar2<scalar_t>{0, 0};
            return SubdomainWaveHoltz<scalar_t>{*this, ab.x, ab.y};
        }
    };

    template <typename scalar_t>
    class DDWaveHoltz
    {
    public:
        DeviceDDWaveHoltz<scalar_t> to_device() const
        {
            return DeviceDDWaveHoltz<scalar_t>{.nt = nt,
                                               .omega = omega,
                                               .weight = weight,
                                               .shift = shift,
                                               .sigma = sigma,
                                               .C = C,
                                               .S = S,
                                               .alpha_beta = reshape(alpha_beta.device_read(), mx_ndof, n_domains)};
        }

        template <typename T>
        friend DDWaveHoltz<T> make_DDWaveHoltz_3d(T omega, const double *a, const H1Space3D &fem,
                                                  const EnsembleSpace3D &efem);

        template <typename T>
        friend DDWaveHoltz<T> make_DDWaveHoltz_2d(T omega, const double *a, const H1Space2D &fem,
                                                  const EnsembleSpace &efem);

    private:
        int n_domains;
        int mx_ndof;

        int nt;
        scalar_t omega;

        scalar_t weight;
        scalar_t shift;

        scalar_t sigma;

        scalar_t C, S; // cos(omega * dt / 2), sin(omega * dt / 2)

        HostDeviceArray<cuddh::scalar2<scalar_t>> alpha_beta;
    };
} // namespace cuddh
