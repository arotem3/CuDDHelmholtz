#ifndef CUDDH_FACE_LINEAR_FUNCTIONAL_HPP
#define CUDDH_FACE_LINEAR_FUNCTIONAL_HPP

#include "H1Space.hpp"
#include "HostDeviceArray.hpp"
#include "forall.hpp"

namespace cuddh
{
    /// @brief computes inner products (f, phi) for face data f = f(x)
    /// (x on faces defined by FaceSpace)
    class FaceLinearFunctional
    {
    public:
        FaceLinearFunctional(const FaceSpace& fs);
        FaceLinearFunctional(const FaceSpace& fs, const QuadratureRule& quad);

        /// @brief F[i] <- F[i] + c * (f, phi[i])
        /// where f=f(x) and phi[i] is the i-th basis function in the FaceSpace.
        /// @tparam Func invocable as (const double x[2]) -> double
        /// @param c scalar coefficient
        /// @param f f(const double x[2]) -> double
        /// @param F has length of fs.size(); On exit, F[i] <- F[i] + c * (f, phi[i])
        template <typename Func>
        void action(double c, Func && f, double * F) const;

    private:
        const FaceSpace& fs;
        const Mesh2D::EdgeMetricCollection& metrics;

        const int n_faces;
        const int n_basis;
        const int n_quad;

        const bool fast;

        host_device_dvec _w;
        host_device_dvec _P;
    };

    template <int NQ, typename Func>
    void fl_action(Func && f,
                   int n_faces,
                   int n_basis,
                   int n_quad,
                   const double * __restrict__ d_w,
                   const double * __restrict__ d_P,
                   const double * __restrict__ d_detJ,
                   const double * __restrict__ d_X,
                   const int * __restrict__ d_I,
                   double c,
                   double * __restrict__ d_F)
    {
        auto w = reshape(d_w, n_quad);
        auto P = reshape(d_P, n_quad, n_basis);
        auto detJ = reshape(d_detJ, n_quad, n_faces);
        auto X = reshape(d_X, 2, n_quad, n_faces);
        auto I = reshape(d_I, n_basis, n_faces);

        forall_1d(n_quad, n_faces, [=] __device__ (int e) -> void
        {
            __shared__ double g[NQ];

            const int j = threadIdx.x;

            // evaluate on quadrature points and scale by w and jacobian
            double xi[2];
            xi[0] = X(0, j, e);
            xi[1] = X(1, j, e);
            
            double fi = f(xi);
            g[j] = fi * w(j) * detJ(j, e);

            __syncthreads();

            // integrate
            if (j < n_basis)
            {
                double qg = 0.0;
                for (int i = 0; i < n_quad; ++i)
                    qg += P(i, j) * g[i];
                qg *= c;

                const int idx = I(j, e);
                atomicAdd(d_F + idx, qg);
            }
        });
    }

    template <typename Func>
    void fl_fast(Func && f,
                 int n_faces,
                 int n_basis,
                 const double * __restrict__ d_w,
                 const double * __restrict__ d_detJ,
                 const double * __restrict__ d_X,
                 const int * __restrict__ d_I,
                 double c,
                 double * __restrict__ d_F)
    {
        auto w = reshape(d_w, n_basis);
        auto detJ = reshape(d_detJ, n_basis, n_faces);
        auto X = reshape(d_X, 2, n_basis, n_faces);
        auto I = reshape(d_I, n_basis, n_faces);

        forall_1d(n_basis, n_faces, [=] __device__ (int e) -> void 
        {
            const int k = threadIdx.x;

            double xi[2];
            xi[0] = X(0, k, e);
            xi[1] = X(1, k, e);

            double fi = f(xi);
            fi *= c * w(k) * detJ(k, e);

            const int idx = I(k, e);
            atomicAdd(d_F + idx, fi);
        });
    }

    template <typename Func>
    void FaceLinearFunctional::action(double c, Func && f, double * F) const
    {
        const double * d_w = _w.device_read();
        const double * d_P = _P.device_read();

        const double * d_detJ = metrics.measures(MemorySpace::DEVICE);
        const double * d_X = metrics.physical_coordinates(MemorySpace::DEVICE);

        const int * d_I = fs.subspace_indices(MemorySpace::DEVICE);

        if (fast)
            fl_fast(f, n_faces, n_basis, d_w, d_detJ, d_X, d_I, c, F);
        else
        {
            if (n_quad <= 4)
                fl_action<4>(f, n_faces, n_basis, n_quad, d_w, d_P, d_detJ, d_X, d_I, c, F);
            else if (n_quad <= 8)
                fl_action<8>(f, n_faces, n_basis, n_quad, d_w, d_P, d_detJ, d_X, d_I, c, F);
            else if (n_quad <= 12)
                fl_action<12>(f, n_faces, n_basis, n_quad, d_w, d_P, d_detJ, d_X, d_I, c, F);
            else if (n_quad <= 16)
                fl_action<16>(f, n_faces, n_basis, n_quad, d_w, d_P, d_detJ, d_X, d_I, c, F);
            else if (n_quad <= 24)
                fl_action<24>(f, n_faces, n_basis, n_quad, d_w, d_P, d_detJ, d_X, d_I, c, F);
            else if (n_quad <= 32)
                fl_action<32>(f, n_faces, n_basis, n_quad, d_w, d_P, d_detJ, d_X, d_I, c, F);
            else if (n_quad <= 64)
                fl_action<64>(f, n_faces, n_basis, n_quad, d_w, d_P, d_detJ, d_X, d_I, c, F);
            else
                cuddh_error("FaceLinearFunctional::action does not support quadrature points with more than 64 points");
        }
    }

} // namespace cuddh

#endif
