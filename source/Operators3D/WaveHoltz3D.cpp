#include "Operators3D/WaveHoltz3D.hpp"

#include <thrust/extrema.h>

#include <algorithm>

#include "Operators3D/FaceMassMatrix3D.hpp"
#include "forall.hpp"
#include "linalg.hpp"

using namespace cuddh;

static HostDeviceArray<double2> make_alpha_beta(const H1Space3D &fem, double theta, double sigma, const MassMatrix3D &M,
                                                const FaceMassMatrix3D &H)
{
    const int ndof = fem.size();

    auto m = M.to_device();
    auto h = H.to_device();

    HostDeviceArray<double2> _ab(ndof);
    auto ab = reshape(_ab.device_write(), ndof);

    forall(ndof, [=] __device__(int i) mutable {
        double mi = m(i), hi = h(i);

        double inv = 1 / (mi + theta * hi);

        double2 alpha_beta;
        alpha_beta.x = (mi - theta * hi) * inv;
        alpha_beta.y = sigma * inv;

        ab(i) = alpha_beta;
    });

    return _ab;
}

static GridFunc3D<double> square(const GridFunc3D<double> &a)
{
    return a.transform([] __device__(double x) -> double { return x * x; });
}

static double compute_nt(double omega, double h, double p, const GridFunc3D<double> *a)
{
    double c = 1.0;

    if (a)
    {
        auto aview = a->read(MemorySpace::HOST);
        c = *std::min_element(aview.begin(), aview.end());
    }

    cuddh_verify(
        c > 0,
        printf("WaveHoltz3D error: coefficient a must be strictly positive. Encountered non-positive value: %f.\n", c));

    double dt = 2.0 * h * c / (p * p);
    double T = 2.0 * M_PI / omega;
    return std::max<int>(std::ceil(T / dt), 5);
}

WaveHoltz3D::WaveHoltz3D(const H1Space3D &fem, const TraceSpace3D &fs, double omega)
    : Operator<double>(2 * fem.size()), omega(omega), stiffness(fem), acc(fem.size()), w(this->ndof())
{
    nt = compute_nt(omega, fem.mesh().h(), fem.basis().size(), nullptr);

    double tan_val = std::tan(M_PI / nt);
    shift = 0.25 - 0.25 * tan_val * tan_val;

    const double theta = std::tan(M_PI / nt) / omega;
    const double sigma = std::sin(M_PI / nt) / (0.5 * omega);

    MassMatrix3D M(fem);
    FaceMassMatrix3D H(fs);
    ab = make_alpha_beta(fem, theta, sigma, M, H);
}

WaveHoltz3D::WaveHoltz3D(const H1Space3D &fem, const TraceSpace3D &fs, double omega, const GridFunc3D<double> &a)
    : Operator<double>(2 * fem.size()), omega(omega), stiffness(fem), acc(fem.size()), w(this->ndof())
{
    nt = compute_nt(omega, fem.mesh().h(), fem.basis().size(), &a);

    double tan_val = std::tan(M_PI / nt);
    shift = 0.25 - 0.25 * tan_val * tan_val;

    const double theta = std::tan(M_PI / nt) / omega;
    const double sigma = std::sin(M_PI / nt) / (0.5 * omega);

    MassMatrix3D M(fem, square(a));
    FaceMassMatrix3D H(fs, a);
    ab = make_alpha_beta(fem, theta, sigma, M, H);
}

void WaveHoltz3D::action(double c, const double *x, double *y) const
{
    dla::axpby(this->ndof(), c, x, 1.0, y); // y <- y + c * x
    S(-c, x, y);                            // y <- y - c * S(x) = y + c * (I - S) * x
}

void WaveHoltz3D::action(const double *x, double *y) const
{
    dla::copy(this->ndof(), x, y); // y <- x
    S(-1.0, x, y);                 // y <- y - S(x) = y + (I - S) * x
}

void WaveHoltz3D::evolve_project(double C, const double *d_u, const double *d_f, double *d_out) const
{
    if (not d_u && not d_f)
        return;

    cuddh_verify(d_out != nullptr, printf("WaveHoltz3D::evolve_project: d_out is null"));

    const int ndof = this->ndof() / 2;
    const double sigma = std::sin(M_PI / nt) / (0.5 * omega);

    double *d_w = w.device_write();
    double *p = d_w;
    double *q = d_w + ndof;
    double *out_u = d_out;
    double *out_v = d_out + ndof;

    const double *b = (d_f) ? d_f : nullptr;
    const double *c = (d_f) ? d_f + ndof : nullptr;

    auto alpha_beta = reshape(ab.device_read(), ndof);

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
        // update p
        Ku = C * filter(n);
        forall(ndof, [=] __device__(int i) -> void {
            p[i] += sigma * q[i];
            out_u[i] += Ku * p[i];
        });

        // update q
        stiffness.action(p, a);

        double cs = std::cos(2.0 * M_PI * n / nt);
        double sn = std::sin(2.0 * M_PI * n / nt);
        Kv = C * filter(n + 0.5) / omega;
        forall(ndof, [=] __device__(int i) -> void {
            const auto [alpha, beta] = alpha_beta(i);

            double acc_i = -a[i];
            if (d_f)
                acc_i += cs * b[i] + sn * c[i];

            q[i] = alpha * q[i] + beta * acc_i;
            out_v[i] += Kv * q[i];
        });
    }
}
