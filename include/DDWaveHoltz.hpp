#pragma once

#include "DD2D/EnsembleSpace.hpp"
#include "DD3D/EnsembleSpace3D.hpp"
#include "FEM2D/GridFunc2D.hpp"
#include "cxmult.hpp"

namespace cuddh
{
    template <typename scalar_t>
    struct DeviceDDWaveHoltz;

    template <typename scalar_t, int TDOF = 1>
    struct SubdomainWaveHoltz
    {
        using vec_t = scalar2<scalar_t>;
        using arr_t = cuda::std::array<vec_t, TDOF>;

        const DeviceDDWaveHoltz<scalar_t> &wh;
        scalar_t alpha[TDOF], beta[TDOF];

        template <typename SubdomainStiffnessMatrix>
        __forceinline__ __device__ vec_t operator()(const SubdomainStiffnessMatrix &A, vec_t u, const vec_t &F) const
            requires(TDOF == 1)
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

                q = alpha[0] * q + beta[0] * (-A(p) + cs * F.x + sn * F.y);

                details::cxmult(cs, sn, wh.C, wh.S);
                K = wh.filter(cs);

                u.y += K * q;
            }

            u.y /= wh.omega;

            return u;
        }

        template <typename SubdomainStiffnessMatrix>
        __forceinline__ __device__ arr_t operator()(const SubdomainStiffnessMatrix &A, arr_t u, const arr_t &F) const
        {
            scalar_t cs = 1, sn = 0;
            scalar_t K = wh.filter(cs);

            cuda::std::array<scalar_t, TDOF> p, q;

#pragma unroll TDOF
            for (int t = 0; t < TDOF; ++t)
                p[t] = u[t].x;

            details::cxmult(cs, sn, wh.C, wh.S);

#pragma unroll TDOF
            for (int t = 0; t < TDOF; ++t)
                q[t] = (-sn * u[t].x + cs * u[t].y) * wh.omega;

#pragma unroll TDOF
            for (int t = 0; t < TDOF; ++t)
                u[t].x = K * p[t];

            K = wh.filter(cs);

#pragma unroll TDOF
            for (int t = 0; t < TDOF; ++t)
                u[t].y = K * q[t];

            for (int n = 1; n < wh.nt; ++n)
            {
                details::cxmult(cs, sn, wh.C, wh.S);
                K = wh.filter(cs);

#pragma unroll TDOF
                for (int t = 0; t < TDOF; ++t)
                    p[t] += wh.sigma * q[t];

#pragma unroll TDOF
                for (int t = 0; t < TDOF; ++t)
                    u[t].x += K * p[t];

                const auto Ap = A(p);

#pragma unroll TDOF
                for (int t = 0; t < TDOF; ++t)
                    q[t] = alpha[t] * q[t] + beta[t] * (-Ap[t] + cs * F[t].x + sn * F[t].y);

                details::cxmult(cs, sn, wh.C, wh.S);
                K = wh.filter(cs);

#pragma unroll TDOF
                for (int t = 0; t < TDOF; ++t)
                    u[t].y += K * q[t];
            }

            scalar_t rw = 1 / wh.omega;

#pragma unroll TDOF
            for (int t = 0; t < TDOF; ++t)
                u[t].y *= rw;

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

        template <int TDOF = 1>
        __device__ SubdomainWaveHoltz<scalar_t, TDOF> subspace_op(int subsp, int tid, int ndof) const
        {
            SubdomainWaveHoltz<scalar_t, TDOF> out{.wh = *this};

            for (int t = 0; t < TDOF; ++t)
            {
                const int idx = tid + (blockDim.x * blockDim.y) * t;
                const scalar2<scalar_t> ab = (idx < ndof) ? alpha_beta(idx, subsp) : scalar2<scalar_t>{0, 0};

                out.alpha[t] = ab.x;
                out.beta[t] = ab.y;
            }

            return out;
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
        friend DDWaveHoltz<T> make_DDWaveHoltz_2d(const EnsembleSpace &efem, T omega, const GridFunc2D<double> *a);

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
