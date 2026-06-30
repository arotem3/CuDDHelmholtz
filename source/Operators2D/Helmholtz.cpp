#include "Operators2D/Helmholtz.hpp"

using namespace cuddh;

static GridFunc2D<double> square(const GridFunc2D<double> &a)
{
    return a.transform([] __device__(double x) -> double { return x * x; });
}

void Helmholtz::action(const double *x, double *y) const
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

Helmholtz::Helmholtz(const H1Space2D &fem, const TraceSpace2D &fs, double omega)
    : Operator<double>(2 * fem.size()), omega{omega}, S(fem), M(fem), H(fs)
{}

Helmholtz::Helmholtz(const H1Space2D &fem, const TraceSpace2D &fs, double omega, const GridFunc2D<double> &a)
    : Operator<double>(2 * fem.size()), omega{omega}, S(fem), M(fem, square(a)), H(fs, a)
{}

// Assembles the n×n complex Helmholtz matrix: c*(S - omega^2*M - i*omega*H).
bool Helmholtz::assemble(std::complex<double> c, SparseMatrix<double, true> &out) const
{
    using namespace std::complex_literals;
    S.assemble(c, out);
    M.assemble(-c * (omega * omega), out);
    H.assemble(-1.0i * c * omega, out);
    return true;
}
