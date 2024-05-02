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
        int mx_n_lambda;
        int mx_dof;
        int mx_fdof;
        int mx_elem_per_dom;

        double omega;
        double dt;

        host_device_ivec _B;
        host_device_ivec _dualB;
        host_device_ivec _s_lambda;

        HostDeviceArray<float> _D; // differentiation matrix

        DiagInvMassMatrix g_inv_m; // global inverse mass matrix
        
        HostDeviceArray<float> _g_tensor; // geometric factors for stiffness matrix computations
        HostDeviceArray<float> _inv_m; // inverse mass matrix
        HostDeviceArray<float> _m; // mass matrix 
        HostDeviceArray<float> _H; // face mass matrix
        HostDeviceArray<float> _wh_filter; // omega / pi * (cos(omega * t) - 0.25) scaled by quadrature weights
        HostDeviceArray<float> _cs; // cos(omega t) on all half time steps
        HostDeviceArray<float> _sn; // sin(omega t) on all half time steps

        mutable HostDeviceArray<float> _g_lambda; // global lambda vector
        mutable HostDeviceArray<float> _g_update; // global lambda updates

        std::unique_ptr<EnsembleSpace> efem;
    };
} // namespace cuddh

#endif
