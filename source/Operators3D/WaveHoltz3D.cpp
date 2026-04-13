#include "Operators3D/WaveHoltz3D.hpp"

#include <thrust/extrema.h>

#include "forall.hpp"
#include "linalg.hpp"

using namespace cuddh;

static thrust::device_vector<double> make_face_mass(const TraceSpace3D &tr, const double *d_a)
{
    const int n_faces = tr.n_faces();
    const int n_basis = tr.h1_space().basis().size();

    auto &quad = tr.h1_space().basis().quadrature();

    auto w = quad.w(MemorySpace::DEVICE);
    auto x = quad.x(MemorySpace::DEVICE);

    DeviceMesh3D mesh = tr.h1_space().mesh().to_device();

    auto I = tr.subspace_indices(MemorySpace::DEVICE);
    auto J = tr.global_indices(MemorySpace::DEVICE);

    thrust::device_vector<double> H(tr.h1_space().size(), 0);
    auto d_m = thrust::raw_pointer_cast(H.data());

    forall_2d(n_basis, n_basis, n_faces, [=] __device__(int f) mutable -> void {
        const int tr_idx = I(threadIdx.x, threadIdx.y, f);
        const int fem_idx = J(tr_idx);

        __shared__ QuadFace face;
        if (threadIdx.x == 0 && threadIdx.y == 0)
            face = mesh.face(f);

        double value = w(threadIdx.x) * w(threadIdx.y);
        if (d_a)
            value *= d_a[tr_idx];

        __syncthreads();

        value *= face.measure(double2{x(threadIdx.x), x(threadIdx.y)});

        atomicAdd(d_m + fem_idx, value);
    });

    return H;
}

static HostDeviceArray<double2> make_alpha_beta(double theta, double sigma, const double *a2, const double *a,
                                                const H1Space3D &fem, const TraceSpace3D &fs)
{
    const int ndof = fem.size();

    MassMatrix3D M(a2, fem);
    auto m = diagonal_mass(M, MemorySpace::DEVICE);

    auto H = make_face_mass(fs, a);
    auto h = reshape(thrust::raw_pointer_cast(H.data()), ndof);

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

WaveHoltz3D::WaveHoltz3D(double omega, const double *a2x, const double *ax, const H1Space3D &fem,
                         const TraceSpace3D &fs)
    : Operator<double>(2 * fem.size()), omega(omega), stiffness(fem), acc(fem.size()), w(this->ndof())
{
    const int n = this->ndof() / 2;

    const double maxvel = [&]() -> double {
        auto iter = thrust::device_pointer_cast(a2x);
        double amin = *thrust::min_element(iter, iter + n);
        return 1.0 / std::sqrt(amin);
    }();

    double T = 2.0 * M_PI / omega;
    double p = fem.basis().size();
    double dt = 2.0 * fem.mesh().h() / (p * p * maxvel); // CFL condition

    nt = std::max(std::ceil(T / dt), 5.0);

    double tan = std::tan(M_PI / nt);
    shift = 0.25 - 0.25 * tan * tan;

    const double theta = std::tan(M_PI / nt) / omega;
    const double sigma = std::sin(M_PI / nt) / (0.5 * omega);

    ab = make_alpha_beta(theta, sigma, a2x, ax, fem, fs);
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
