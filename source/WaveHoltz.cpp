#include "WaveHoltz.hpp"

using namespace cuddh;

WaveHoltz cuddh::init_waveholtz(double omega, double dt)
{
    WaveHoltz W;
    W.omega = omega;

    double T = (2 * M_PI) / omega;
    W.nt = std::ceil(T / dt);
    dt = T / W.nt;
    W.dt = dt;

    double tan_omega_dt = std::tan(0.5 * omega * dt);
    double a0 = 0.25 * (1 - tan_omega_dt * tan_omega_dt); // corrected shift

    W.K1 = dt * (omega / M_PI);
    W.K2 = W.K1 * a0;

    return W;
}
