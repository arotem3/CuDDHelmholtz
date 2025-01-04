#include "WaveHoltz.hpp"

using namespace cuddh;

WaveHoltz cuddh::init_waveholtz(double omega, double dt)
{
    WaveHoltz W;
    W.omega = omega;

    double T = (2 * M_PI) / omega;
    W.nt = std::ceil(T / dt);
    W.dt = T / W.nt;

    W.K.resize(W.nt);
    auto K = reshape(W.K.host_write(), W.nt);

    double tan_omega_dt = std::tan(0.5 * omega * dt);
    double a0 = 0.25 * (1 - tan_omega_dt * tan_omega_dt); // corrected shift

    for (int j = 0; j < W.nt; ++j)
        K(j) = dt * (omega / M_PI) * (std::cos(omega * j * dt) - a0);
    // K(0) *= 0.5;
    // K(W.nt) *= 0.5;

    W.cs.resize(2 * W.nt - 1);
    W.sn.resize(2 * W.nt - 1);

    auto cs = reshape(W.cs.host_write(), 2 * W.nt - 1);
    auto sn = reshape(W.sn.host_write(), 2 * W.nt - 1);

    for (int j = 0; j < 2 * W.nt; ++j)
    {
        double t = 0.5 * j * dt;
        cs(j) = -std::cos(omega * t);
        sn(j) = std::sin(omega * t);
    }

    return W;
}
