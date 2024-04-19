#ifndef CUDDH_FACE_LINEAR_FUNCTIONAL_HPP
#define CUDDH_FACE_LINEAR_FUNCTIONAL_HPP

#include "H1Space.hpp"

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
        void action(double c, Func f, double * F) const;



    private:
        const int n_faces;
        const int n_basis;

        const QuadratureRule& quad;
        const bool fast;

        const_dmat_wrapper detJ;
        const_dcube_wrapper x;
        const_imat_wrapper I; // restricted subspace indices

        dmat P;
    };

    template <typename Func>
    void FaceLinearFunctional::action(double c, Func f, double * F) const
    {
        int n_quad = quad.size();
    
        dvec g(n_quad);

        for (int e = 0; e < n_faces; ++e)
        {
            // evaluate on quadrature points and scale by w and jacobian
            for (int i = 0; i < n_quad; ++i)
            {
                double xi[2];
                xi[0] = x(0, i, e);
                xi[1] = x(1, i, e);

                g(i) = quad.w(i) * detJ(i, e) * f(xi);
            }

            if (fast)
            {
                for (int i = 0; i < n_basis; ++i)
                    F[I(i, e)] += c * g(i);
            }
            else
            {
                // integrate
                for (int k = 0; k < n_basis; ++k)
                {
                    double qg = 0.0;
                    for (int i = 0; i < n_quad; ++i)
                        qg += P(i, k) * g(i);
                    F[I(k, e)] += c * qg;
                }
            }
        }
    }

} // namespace cuddh


#endif