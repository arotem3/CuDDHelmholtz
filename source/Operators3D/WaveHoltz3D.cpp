#include "Operators3D/WaveHoltz3D.hpp"

using namespace cuddh;

static void init_face_mass(const TraceSpace3D &tr, const double *d_a, double *d_m)
{
    const int n_faces = tr.n_faces();
    const int n_basis = tr.h1_space().basis().size();

    auto &quad = tr.h1_space().basis().quadrature();

    host_device_dvec _w(n_basis);
    double *h_w = _w.host_write();
    for (int i = 0; i < n_basis; ++i)
        h_w[i] = quad.w(i);
    auto w = reshape(_w.device_read(), n_basis);

    host_device_dvec _x(n_basis);
    double *h_x = _x.host_write();
    for (int i = 0; i < n_basis; ++i)
        h_x[i] = quad.x(i);
    auto x = reshape(_x.device_read(), n_basis);

    DeviceMesh3D mesh = tr.h1_space().mesh().to_device();

    auto I = tr.subspace_indices(MemorySpace::DEVICE);
    auto J = tr.global_indices(MemorySpace::DEVICE);

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
}

WaveHoltz3D::WaveHoltz3D(double omega, double maxvel, const double *a2x, const double *ax, const H1Space3D &fem_,
                         const TraceSpace3D &fs_)
    : omega(omega),
      ndof(fem_.size()),
      fem(fem_),
      fs(fs_),
      stiffness(fem_),
      M(a2x, fem_),
      H(ndof),
      acc(ndof),
      w(2 * ndof)
{
    init_face_mass(fs, ax, H.device_write());

    double T = 2.0 * M_PI / omega;
    double p = fem.basis().size();
    double dt = 2.0 * fem.mesh().h() / (p * p * maxvel); // CFL condition

    nt = std::max(std::ceil(T / dt), 5.0);
    dt = T / nt;

    double tan = std::tan(M_PI / nt);
    shift = 0.25 - 0.25 * tan * tan;
}

void WaveHoltz3D::action(double c, const double *x, double *y) const
{
    dla::axpby(2 * ndof, c, x, 1.0, y); // y <- y + c * x
    S(-c, x, y);                        // y <- y - c * S(x) = y + c * (I - S) * x
}

void WaveHoltz3D::action(const double *x, double *y) const
{
    dla::copy(2 * ndof, x, y); // y <- x
    S(-1.0, x, y);             // y <- y - S(x) = y + (I - S) * x
}

void WaveHoltz3D::evolve_project(double C, const double *d_u, const double *d_f, double *d_out) const
{
    if (not d_u && not d_f)
        return;

    cuddh_verify(d_out != nullptr, printf("WaveHoltz3D::evolve_project: d_out is null"));

    const int ndof = this->ndof;
    const double theta = std::tan(M_PI / nt) / omega;
    const double sigma = std::sin(M_PI / nt) / (0.5 * omega);

    double *d_w = w.device_write();
    double *p = d_w;
    double *q = d_w + ndof;
    double *out_u = d_out;
    double *out_v = d_out + ndof;

    const double *b = (d_f) ? d_f : nullptr;
    const double *c = (d_f) ? d_f + ndof : nullptr;

    auto m = diagonal_mass(M, MemorySpace::DEVICE);
    auto h = reshape(H.device_read(), ndof);

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
            double mi = m[i];
            double hi = h[i];

            double inv = 1.0 / (mi + theta * hi);
            double alpha = (mi - theta * hi) * inv;
            double beta = sigma * inv;

            double acc_i = -a[i];
            if (d_f)
                acc_i += cs * b[i] + sn * c[i];

            q[i] = alpha * q[i] + beta * acc_i;
            out_v[i] += Kv * q[i];
        });
    }
}
