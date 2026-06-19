#include "Operators3D/Helmholtz3D.hpp"

#include "forall.hpp"

using namespace cuddh;

static GridFunc3D<double> square(const GridFunc3D<double> &a)
{
    return a.transform([] __device__(double x) -> double { return x * x; });
}

Helmholtz3D::Helmholtz3D(const H1Space3D &fem, const TraceSpace3D &tr, double omega)
    : Operator<double>(2 * fem.size()), omega{omega}, S(fem), M(fem), H(tr)
{}

Helmholtz3D::Helmholtz3D(const H1Space3D &fem, const TraceSpace3D &tr, double omega, const GridFunc3D<double> &a)
    : Operator<double>(2 * fem.size()), omega{omega}, S(fem), M(fem, square(a)), H(tr, a)
{}

void Helmholtz3D::action(const double *x, double *y) const
{
    const int n = this->ndof() / 2;

    const double *u = x;
    const double *v = x + n;

    double *Au = y;
    double *Av = y + n;

    S.action(u, Au);
    S.action(v, Av);

    double omega = this->omega;
    double om2 = omega * omega;
    auto m = M.to_device();
    auto h = H.to_device();

    forall(n, [=] __device__(int i) {
        const double mi = om2 * m(i);
        const double hi = omega * h(i);

        const double U = u[i], V = v[i];

        Au[i] = Au[i] - mi * U + hi * V;
        Av[i] = -Av[i] + mi * V + hi * U;
    });
}
