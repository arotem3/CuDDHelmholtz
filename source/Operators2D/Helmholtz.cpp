#include "Operators2D/Helmholtz.hpp"

using namespace cuddh;

void Helmholtz::action(const double *x, double *y) const
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
        const double mi = m(i);
        const double hi = h(i);

        const double U = u[i], V = v[i];

        Au[i] = Au[i] - omega * omega * mi * U + omega * hi * V;
        Av[i] = -Av[i] + omega * omega * mi * V + omega * hi * U;
    });
}

Helmholtz::Helmholtz(double omega_, const double *a2x, const double *ax, const H1Space2D &fem, const TraceSpace2D &fs)
    : omega{omega_}, ndof{fem.size()}, S(fem), M(fem, a2x), H(fs, ax)
{}
