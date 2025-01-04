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
        int nt;
        float omega;
        float dt;

        float K1; // filter (weighted by dt) = K1 * cos(omega * t) - K2
        float K2;

        constexpr __host__ __device__ float K(float cs) const
        {
            return K1 * cs - K2;
        }
    };

    WaveHoltz init_waveholtz(double omega, double dt);
} // namespace cuddh

#endif