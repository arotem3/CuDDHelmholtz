#ifndef CUDDH_WAVEHOLTZ_HPP
#define CUDDH_WAVEHOLTZ_HPP

#include "cuddh_config.hpp"
#include "cuddh_error.hpp"
#include "HostDeviceArray.hpp"
#include "Tensor.hpp"

namespace cuddh
{
    struct WaveHoltz
    {
        double omega;              // frequency
        double dt;                 // time step
        int nt;                    // number of time steps
        HostDeviceArray<float> K;  // omega / pi * (cos(omega * t) - 0.25) scaled by quadrature weights
        HostDeviceArray<float> cs; // cos(omega t) on all half time steps
        HostDeviceArray<float> sn; // sin(omega t) on all half time steps

        struct KernelWaveHoltz
        {
            float omega;
            float dt;
            int nt;
            VectorWrapper<const float> K;
            VectorWrapper<const float> cs;
            VectorWrapper<const float> sn;
        };

        KernelWaveHoltz to_device() const
        {
            auto Kvec = reshape(K.device_read(), nt);
            auto csvec = reshape(cs.device_read(), 2 * nt - 1);
            auto snvec = reshape(sn.device_read(), 2 * nt - 1);
            return {(float)omega, (float)dt, nt, Kvec, csvec, snvec};
        }
    };

    WaveHoltz init_waveholtz(double omega, double dt);
} // namespace cuddh

#endif