#ifndef CUDDH_DDH_HPP
#define CUDDH_DDH_HPP

#include <cuda_runtime.h>

#include "Operator.hpp"
#include "EnsembleSpace.hpp"
#include "MassMatrix.hpp"
#include "linalg.hpp"

#include "HostDeviceArray.hpp"
#include "forall.hpp"

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
        int mx_n_lambda;
        int mx_dof;
        int mx_fdof;
        int mx_elem_per_dom;

        double omega;
        double dt;

        host_device_ivec _B;
        host_device_ivec _dualB;
        host_device_ivec _s_lambda;

        host_device_dvec _D; // differentiation matrix

        DiagInvMassMatrix g_inv_m; // global inverse mass matrix
        
        host_device_dvec _g_tensor; // geometric factors for stiffness matrix computations
        host_device_dvec _inv_m; // inverse mass matrix
        host_device_dvec _m; // mass matrix 
        host_device_dvec _H; // face mass matrix
        host_device_dvec _wh_filter; // omega / pi * (cos(omega * t) - 0.25) scaled by quadrature weights
        host_device_dvec _cs; // cos(omega t) on all half time steps
        host_device_dvec _sn; // sin(omega t) on all half time steps

        mutable host_device_dvec _g_lambda; // global lambda vector
        mutable host_device_dvec _g_update; // global lambda updates

        std::unique_ptr<EnsembleSpace> efem;
    };
} // namespace cuddh


#endif