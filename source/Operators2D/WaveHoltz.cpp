#include "Operators2D/WaveHoltz.hpp"

using namespace cuddh;

WaveHoltz::WaveHoltz(double omega, double maxvel, const double *a2x, const double *ax, const H1Space2D &fem,
                     const TraceSpace2D &fs)
    : Operator<double>(2 * fem.size()),
      omega(omega),
      stiffness(fem),
      mass(fem, a2x),
      face_mass(fs, ax),
      acc(fem.size()),
      w(this->ndof())
{
    double T = 2.0 * M_PI / omega;
    double p = fem.basis().size();
    double dt = 2.0 * fem.mesh().h() / (p * p * maxvel); // CFL condition

    nt = std::max(std::ceil(T / dt), 5.0);
    dt = T / nt;

    double tan = std::tan(M_PI / nt);
    shift = 0.25 - 0.25 * tan * tan;
}

void WaveHoltz::action(double c, const double *x, double *y) const
{
    dla::axpby(this->ndof(), c, x, 1.0, y); // y <- y + c * x
    S(-c, x, y);                            // y <- y - c * S(x) = y + c * (I - S) * x
}

void WaveHoltz::action(const double *x, double *y) const
{
    dla::copy(this->ndof(), x, y); // y <- x
    S(-1.0, x, y);                 // y <- y - S(x) = x - S(x)
}

void WaveHoltz::evolve_project(double C, const double *d_u, const double *d_f, double *d_out) const
{
    if (not d_u && not d_f)
        return;

    cuddh_verify(d_out != nullptr, printf("WaveHoltz::evolve_project: d_out is null"));

    const int ndof = this->ndof() / 2;
    const double theta = std::tan(M_PI / nt) / omega;
    const double sigma = std::sin(M_PI / nt) / (0.5 * omega);

    double *d_w = w.device_write();
    double *p = d_w;
    double *q = d_w + ndof;
    double *out_u = d_out;
    double *out_v = d_out + ndof;

    const double *b = (d_f) ? d_f : nullptr;
    const double *c = (d_f) ? d_f + ndof : nullptr;

    auto m = mass.to_device();
    auto h = face_mass.to_device();

    double *a = acc.device_write();

    double omega_cs = omega * std::cos(M_PI / nt);
    double omega_sn = omega * std::sin(M_PI / nt);
    double Ku = C * filter(0);
    double Kv = C * filter(0.5) / omega;

    forall(ndof, [=] __device__(int i) -> void {
        double u = (d_u) ? d_u[i] : 0.0;
        double v = (d_u) ? d_u[i + ndof] : 0.0;

        p[i] = u;
        q[i] = -omega_sn * u + omega_cs * v;

        if (d_u)
        {
            out_u[i] += Ku * p[i];
            out_v[i] += Kv * q[i];
        }
    });

    for (int n = 1; n < nt; ++n)
    {
        // update u, p
        Ku = C * filter(n);
        forall(ndof, [=] __device__(int i) -> void {
            p[i] += sigma * q[i];
            out_u[i] += Ku * p[i];
        });

        // update v, q
        stiffness.action(p, a);

        double cs = std::cos(2 * M_PI * n / nt);
        double sn = std::sin(2 * M_PI * n / nt);
        Kv = C * filter(n + 0.5) / omega;
        forall(ndof, [=] __device__(int i) -> void {
            double M = m(i);
            double H = h(i);

            double inv = 1.0 / (M + theta * H);
            double alpha = (M - theta * H) * inv;
            double beta = sigma * inv;

            double acc = -a[i];
            if (d_f)
                acc += cs * b[i] + sn * c[i];

            q[i] = alpha * q[i] + beta * acc;
            out_v[i] += Kv * q[i];
        });
    }
}
