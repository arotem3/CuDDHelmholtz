#ifndef CUDDH_HELMHOLTZ_HPP
#define CUDDH_HELMHOLTZ_HPP

#include <thrust/universal_vector.h>

#include "cuddh.hpp"

using namespace cuddh;

/// @brief FEM discretization of the Helmholtz equation -div(grad u) - omega^2 u == f
/// with boundary conditions: du/dn + i omega u == 0.
class Helmholtz : public Operator
{
public:
    Helmholtz(double omega_, const double *a2x, const double *ax, const H1Space2D &fem, const TraceSpace2D &fs)
        : omega{omega_}, ndof{fem.size()}, S(fem), M(fem, a2x), H(fs, ax)
    {}

    /// @brief y[i] = a(x, phi[i]) where a(u,v) = (grad u, grad v) - omega^2 (u, v) - i*omega <u, v>
    /// @param x the real and imaginary part of the solution
    /// @param y on exit y[i] <- a(x, v[i]) for each v
    void action(const double *x, double *y) const
    {
        const double *u = x;
        const double *v = x + ndof;

        double *Au = y;
        double *Av = y + ndof;

        S.action(u, Au);
        S.action(v, Av);

        double omega = this->omega;
        auto m = M.to_device();
        auto h = H.to_device();

        forall(ndof, [=] __device__(int i) -> void {
            double mi = m(i);
            double hi = h(i);

            double U = u[i], V = v[i];

            Au[i] = Au[i] - omega * omega * mi * U + omega * hi * V;
            Av[i] = -Av[i] + omega * omega * mi * V + omega * hi * U;
        });
    }

    /// @brief y[i] <- y[i] + c * a(x, phi[i]) where a(u,v) = (grad u, grad v) - omega^2 (u, v) - i*omega <u, v>
    /// @param c
    /// @param x the real and imaginary part of the solution
    /// @param y on exit y[i] <- a(x, v[i]) for each v
    void action(double c, const double *x, double *y) const
    {
        cuddh_assert(false, printf("Helmholtz::action(c, x, y) not implemented\n"));
    }

private:
    const double omega;
    const int ndof;

    StiffnessMatrix S;
    MassMatrix M;
    FaceMassMatrix H;
};

#endif