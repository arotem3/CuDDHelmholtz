#ifndef CUDDH_LINEAR_FUNCTIONAL_HPP
#define CUDDH_LINEAR_FUNCTIONAL_HPP

#include "H1Space.hpp"

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
        void action(double c, Func f, double * F) const;

    private:
        const int n_elem;
        const int n_basis;

        const QuadratureRule& quad;
        const bool fast;

        TensorWrapper<3, const double> detJ;
        TensorWrapper<4, const double> x;
        const_icube_wrapper I;

        dmat P;
    };

    template <typename Func>
    void LinearFunctional::action(double c, Func f, double * F) const
    {
        int n_quad = quad.size();

        dmat g(n_quad, n_quad);
        dmat Pg(n_quad, n_basis);

        for (int el = 0; el < n_elem; ++el)
        {
            for (int j = 0; j < n_quad; ++j)
            {
                for (int i = 0; i < n_quad; ++i)
                {
                    double xij[2];
                    xij[0] = x(0, i, j, el);
                    xij[1] = x(1, i, j, el);

                    g(i, j) = quad.w(i) * quad.w(j) * detJ(i, j, el) * f(xij);
                }
            }

            if (fast)
            {
                for (int k = 0; k < n_basis; ++k)
                    for (int l = 0; l < n_basis; ++l)
                        F[I(k, l, el)] += c * g(k, l);
            }
            else
            {
                for (int i = 0; i < n_quad; ++i)
                {
                    for (int l = 0; l < n_basis; ++l)
                    {
                        double qu = 0.0;
                        for (int j = 0; j < n_quad; ++j)
                        {
                            qu += P(j, l) * g(i, j);
                        }
                        Pg(i, l) = qu;
                    }
                }

                for (int k = 0; k < n_basis; ++k)
                {
                    for (int l = 0; l < n_basis; ++l)
                    {
                        double qqu = 0.0;
                        for (int i = 0; i < n_quad; ++i)
                        {
                            qqu += P(i, k) * Pg(i, l);
                        }
                        F[I(k, l, el)] += c * qqu;
                    }
                }
            }
        }
    }
} // namespace cuddh

#endif
