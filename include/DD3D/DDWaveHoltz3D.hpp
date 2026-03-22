#pragma once

#include "DDFaceMassMatrix3D.hpp"
#include "DDMassMatrix3D.hpp"
#include "DDStiffnessMatrix3D.hpp"
#include "EnsembleSpace3D.hpp"
#include "HostDeviceArray.hpp"
#include "LinearSolvers/gcro.hpp"
#include "Operator.hpp"
#include "Operators3D/MassMatrix3D.hpp"
#include "cuddh_config.hpp"
#include "cuddh_error.hpp"
#include "forall.hpp"
#include "linalg.hpp"

namespace cuddh
{
    namespace details
    {
        // computes the complex multiplication (c + i*s) * (x + i*y) and stores the result in x and y.
        template <typename scalar_t>
        constexpr static void cxmult(scalar_t &x, scalar_t &y, scalar_t c, scalar_t s)
        {
            scalar_t t = x;
            x = c * t - s * y;
            y = s * t + c * y;
        }
    } // namespace details

    template <typename scalar_t>
    struct DeviceDDWaveHoltz3D;

    template <typename scalar_t>
    struct SubdomainWaveHoltz3D
    {
        const DeviceDDWaveHoltz3D<scalar_t> &wh;
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
    struct DeviceDDWaveHoltz3D
    {
        int nt;
        scalar_t omega;

        scalar_t weight;
        scalar_t shift;

        scalar_t sigma;

        scalar_t C, S; // cos(omega * dt / 2), sin(omega * dt / 2)

        MatrixWrapper<const scalar2<scalar_t>> alpha_beta;

        inline constexpr scalar_t filter(scalar_t cs) const { return weight * cs - shift; }

        __device__ SubdomainWaveHoltz3D<scalar_t> subspace_op(int subsp, int i, int ndof) const
        {
            const scalar2<scalar_t> ab = (i < ndof) ? alpha_beta(i, subsp) : scalar2<scalar_t>{0, 0};
            return SubdomainWaveHoltz3D<scalar_t>{*this, ab.x, ab.y};
        }
    };

    template <typename scalar_t>
    class DDWaveHoltz3D
    {
    public:
        DDWaveHoltz3D(scalar_t omega, const double *a, const H1Space3D &fem, const EnsembleSpace3D &efem);

        DeviceDDWaveHoltz3D<scalar_t> to_device() const
        {
            return DeviceDDWaveHoltz3D<scalar_t>{.nt = nt,
                                                 .omega = omega,
                                                 .weight = weight,
                                                 .shift = shift,
                                                 .sigma = sigma,
                                                 .C = C,
                                                 .S = S,
                                                 .alpha_beta = reshape(alpha_beta.device_read(), mx_ndof, n_domains)};
        }

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

    extern template class DDWaveHoltz3D<float>;
    extern template class DDWaveHoltz3D<double>;
} // namespace cuddh
