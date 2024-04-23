#ifndef CUDDH_LINEAR_FUNCTIONAL_HPP
#define CUDDH_LINEAR_FUNCTIONAL_HPP

#include "H1Space.hpp"
#include "forall.hpp"

namespace cuddh
{
    /// @brief computes inner product (f, phi) for function f = f(x)
    class LinearFunctional
    {
    public:
        LinearFunctional(const H1Space& fem);
        LinearFunctional(const H1Space& fem, const QuadratureRule& quad);

        /// @brief F[i] <- F[i] + c * (f, phi[i]) where f=f(x) and phi[i] is the i-th basis function in the H1Space.
        /// @tparam Func invocable as (const double x[2]) -> double
        /// @param[in] c scalar coefficient
        /// @param[in] f f(const double x[2]) -> double 
        /// @param[in,out] F has length of fem.size(); On exit F[i] <- F[i] + c * (f, phi[i])
        /// where phi[i] is the i-th basis function in the H1Space fem.
        template <typename Func>
        void action(double c, Func && f, double * F) const;

    private:
        const H1Space& fem;
        
        const int n_elem;
        const int n_basis;
        const int n_quad;

        const bool fast;

        const Mesh2D::ElementMetricCollection& metrics;

        host_device_dvec _w;
        host_device_dvec _P;
    };

    template <int NQ, typename Func>
    void lf_action(Func && f,
                   int n_elem,
                   int n_quad,
                   int n_basis,
                   const double * d_w,
                   const double * d_P,
                   const double * d_detJ,
                   const double * d_X,
                   const int * d_I,
                   double c,
                   double * d_F)
    {
        auto w = reshape(w, n_quad);
        auto _P = reshape(d_P, n_quad, n_basis);
        auto detJ =reshape(d_detJ, n_quad, n_quad, n_elem);
        auto X = reshape(d_X, 2, n_quad, n_quad, n_elem);
        auto I = reshape(d_I, n_basis, n_basis, n_elem);

        forall_2d(n_quad, n_quad, n_elem, [=] __device__ (int el) -> void
        {
            __shared__ double g[NQ][NQ];
            __shared__ double Pg[NQ][NQ];
            __shared__ double P[NQ][NQ];

            const int tx = threadIdx.x;
            const int dx = blockDim.x;
            const int ty = threadIdx.y;
            const int dy = blockDim.y;

            // copy P
            for (int i = tx; i < n_quad; i += dx)
            {
                for (int k = ty; k < n_basis; k += dy)
                {
                    P[i][k] = _P(i, k);
                }
            }

            // eval f on quadrature points
            for (int i = tx; i < n_quad; i += dx)
            {
                for (int j = ty; j < n_quad; j += dy)
                {
                    double xij[2];
                    xij[0] = X(0, i, j, el);
                    xij[1] = X(1, i, j, el);

                    const double fij = f(xij);

                    g[i][j] = w(i) * w(j) * detJ(i, j, el) * fij;
                }
            }

            __syncthreads();

            for (int i = tx; i < n_quad; i += dx)
            {
                for (int l = ty; l < n_basis; l += dy)
                {
                    double qu = 0.0;
                    for (int j = 0; j < n_quad; ++j)
                    {
                        qu += P[j][l] * g[i][j];
                    }
                    Pg[i][l] = qu;
                }
            }

            __syncthreads();

            for (int k = tx; k < n_basis; k += dx)
            {
                for (int l = ty; l < n_basis; l += dy)
                {
                    double qqu = 0.0;
                    for (int i = 0; i < n_quad; ++i)
                    {
                        qqu += P[i][k] * Pg[i][l];
                    }
                    qqu *= c;
                    
                    const int idx = I(k, l, el);
                    AtomicAdd(d_F + idx, qqu);
                }
            }
        });
    }

    template <typename Func>
    void lf_fast(Func && f,
                 int n_elem,
                 int n_basis,
                 const double * d_w,
                 const double * d_detJ,
                 const double * d_X,
                 const int * d_I,
                 double c,
                 double * d_F)
    {
        auto w = reshape(w, n_basis);
        auto detJ = reshape(d_detJ, n_basis, n_basis, n_elem);
        auto X = reshape(d_X, 2, n_basis, n_basis, n_elem);
        auto I = reshape(d_I, n_basis, n_basis, n_elem);

        forall_2d(n_basis, n_basis, n_elem, [=] __device__ (int el) -> void
        {
            const int i = threadIdx.x;
            const int j = threadIdx.y;

            double xij[2];
            xij[0] = X(0, i, j, el);
            xij[1] = X(1, i, j, el);

            double fij = f(xij);
            fij *= c * w(i) * w(j) * detJ(i, j, el);

            const int idx = I(i, j, el);

            AtomicAdd(d_F + idx, fij);
        });
    }

    template <typename Func>
    void LinearFunctional::action(double c, Func && f, double * F) const
    {
        const double * d_w = _w.device_read();
        const double * d_P = _P.device_read();

        const double * d_detJ = metrics.measures(MemorySpace::DEVICE);
        const double * d_X = metrics.physical_coordinates(MemorySpace::DEVICE);

        const int * d_I = fem.global_indices(MemorySpace::DEVICE);

        if (fast)
            lf_fast(f, n_elem, n_basis, d_w, d_detJ, d_X, d_I, c, d_F);
        else
        {
            if (n_quad <= 4)
                lf_action<4>(f, n_elem, n_quad, n_basis, d_w, d_P, d_detJ, d_X, d_I, c, d_F);
            else if (n_quad <= 8)
                lf_action<8>(f, n_elem, n_quad, n_basis, d_w, d_P, d_detJ, d_X, d_I, c, d_F);
            else if (n_quad <= 12)
                lf_action<12>(f, n_elem, n_quad, n_basis, d_w, d_P, d_detJ, d_X, d_I, c, d_F);
            else if (n_quad <= 16)
                lf_action<16>(f, n_elem, n_quad, n_basis, d_w, d_P, d_detJ, d_X, d_I, c, d_F);
            else if (n_quad <= 24)
                lf_action<24>(f, n_elem, n_quad, n_basis, d_w, d_P, d_detJ, d_X, d_I, c, d_F);
            else if (n_quad <= 32)
                lf_action<32>(f, n_elem, n_quad, n_basis, d_w, d_P, d_detJ, d_X, d_I, c, d_F);
            else if (n_quad <= 64)
                lf_action<64>(f, n_elem, n_quad, n_basis, d_w, d_P, d_detJ, d_X, d_I, c, d_F);
            else
                cuddh_error("LinearFunctional::action does not support quadrature rules with more than 64 points");
        }
    }
} // namespace cuddh

#endif
