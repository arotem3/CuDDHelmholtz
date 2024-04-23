#ifndef CUDDH_DDH_HPP
#define CUDDH_DDH_HPP

#include "Operator.hpp"
#include "EnsembleSpace.hpp"
#include "MassMatrix.hpp"
#include "linalg.hpp"

namespace cuddh
{
    class DDH : public Operator
    {
    public:
        DDH(double omega, const H1Space& fem, int nx, int ny);

        void action(double c, const double * x, double * y) const override;

        void action(const double * x, double * y) const override;

    private:
        int g_ndof;
        int g_elem;
        int n_basis;
        int n_domains;
        int n_lambda;
        int nt;
        int wh_maxit;
        int lambda_maxit;

        double omega;
        double dt;

        icube B; // B(:, l, p) = {face dof, g_lambda dof}, where l is the local lambda index on subdomain p
        icube dualB;
        ivec s_lambda;

        dmat D; // differentiation matrix

        DiagInvMassMatrix g_inv_m; // global inverse mass matrix
        Tensor<4, double> g_tensor; // geometric factors for stiffness matrix computations
        dmat inv_m; // inverse mass matrix
        dmat m; // mass matrix 
        dmat H; // face mass matrix
        dvec wh_filter; // omega / pi * (cos(omega * t) - 0.25) scaled by quadrature weights
        dvec cs; // cos(omega t) on all half time steps
        dvec sn; // sin(omega t) on all half time steps

        mutable dvec g_lambda; // global lambda vector
        mutable dvec g_update; // global lambda updates

        std::unique_ptr<EnsembleSpace> efem;
    };
} // namespace cuddh


#endif