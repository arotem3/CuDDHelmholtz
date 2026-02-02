#include "Operators2D/WaveHoltz.hpp"

using namespace cuddh;

static void init_mass(const H1Space2D &fem, const double *d_a, double *d_m)
{
    const int n_elem = fem.mesh().n_elem();
    const int n_basis = fem.basis().size();

    auto &q = fem.basis().quadrature();
    auto &metrics = fem.mesh().element_metrics(q);
    auto detJ = reshape(metrics.measures(MemorySpace::DEVICE), n_basis, n_basis, n_elem);
    auto I = fem.global_indices(MemorySpace::DEVICE);

    host_device_dvec _w(n_basis);
    double *h_w = _w.host_write();
    for (int i = 0; i < n_basis; ++i)
        h_w[i] = q.w(i);
    auto w = reshape(_w.device_read(), n_basis);

    forall_2d(n_basis, n_basis, n_elem, [=] __device__(int el) mutable -> void
    {
        const int i = threadIdx.x;
        const int j = threadIdx.y;

        const int idx = I(i, j, el);
        double m = w(i) * w(j) * detJ(i, j, el);
        if (d_a)
            m *= d_a[idx];

        atomicAdd(d_m + idx, m);
    });
}

static void init_face_mass(const TraceSpace2D &fs, const double *d_a, double *d_m)
{
    const int n_faces = fs.n_faces();
    const int n_basis = fs.h1_space().basis().size();

    auto &q = fs.h1_space().basis().quadrature();
    auto &metrics = fs.metrics(q);
    auto detJ = reshape(metrics.measures(MemorySpace::DEVICE), n_basis, n_faces);

    auto I = fs.subspace_indices(MemorySpace::DEVICE);
    auto K = fs.global_indices(MemorySpace::DEVICE);

    host_device_dvec _w(n_basis);
    double *h_w = _w.host_write();
    for (int i = 0; i < n_basis; ++i)
        h_w[i] = q.w(i);
    auto w = reshape(_w.device_read(), n_basis);

    forall_1d(n_basis, n_faces, [=] __device__(int f) mutable -> void
    {
        const int k = threadIdx.x;
        const int fs_idx = I(k, f);
        const int fem_idx = K(fs_idx);

        double a = w(k) * detJ(k, f);
        if (d_a)
            a *= d_a[fs_idx];

        atomicAdd(d_m + fem_idx, a);
    });
}

WaveHoltz::WaveHoltz(double omega, double maxvel, const double *a2x, const double *ax, const H1Space2D &fem_, const TraceSpace2D &fs_)
    : omega(omega),
      ndof(fem_.size()),
      fem(fem_),
      fs(fs_),
      stiffness(fem_),
      M(ndof),
      H(ndof),
      acc(ndof),
      w(2 * ndof)
{
    init_mass(fem, a2x, M.device_write());
    init_face_mass(fs, ax, H.device_write());

    double T = 2.0 * M_PI / omega;
    double p = fem.basis().size();
    dt = 2.0 * fem.mesh().min_h() / (p * p * maxvel); // CFL condition

    nt = std::max(std::ceil(T / dt), 5.0);
    dt = T / nt;

    double tan = std::tan(M_PI / nt);
    shift = 0.25 - 0.25 * tan * tan;
}

void WaveHoltz::action(double c, const double *x, double *y) const
{
    axpby(2 * ndof, c, x, 1.0, y); // y <- y + c * x
    S(-c, x, y);                   // y <- y - c * S(x) = y + c * (I - S) * x
}

void WaveHoltz::action(const double *x, double *y) const
{
    copy(2 * ndof, x, y); // y <- x
    S(-1.0, x, y);        // y <- y - S(x) = x - S(x)
}

void WaveHoltz::evolve_project(double C, const double *d_u, const double *d_f, double *d_out) const
{
    if (not d_u && not d_f)
        return;

    if (not d_out)
        throw std::runtime_error("WaveHoltz::evolve_project: d_out is null");

    const int ndof = this->ndof;
    const double theta = std::tan(M_PI / nt) / omega;
    const double sigma = std::sin(M_PI / nt) / (0.5 * omega);

    double *d_w = w.device_write();
    double *p = d_w;
    double *q = d_w + ndof;
    double *out_u = d_out;
    double *out_v = d_out + ndof;

    double cs = omega * std::cos(0.5 * omega * dt);
    double sn = omega * std::sin(0.5 * omega * dt);
    double K = C * filter(0);
    double Khalf = C * filter(0.5) / omega;

    forall(ndof, [=] __device__(int i) -> void
    {
        double u = (d_u) ? d_u[i] : 0.0;
        double v = (d_u) ? d_u[i + ndof] : 0.0;

        p[i] = u;
        q[i] = -sn * u + cs * v;
        
        if (d_u)
        {
            out_u[i] += K * p[i];
            out_v[i] += Khalf * q[i];
        }
    });

    const double *b = (d_f) ? d_f : nullptr;
    const double *c = (d_f) ? d_f + ndof : nullptr;

    auto m = reshape(M.device_read(), ndof);
    auto h = reshape(H.device_read(), ndof);

    double *a = acc.device_write();

    for (int n = 1; n < nt; ++n)
    {
        // update u
        K = C * filter(n);
        forall(ndof, [=] __device__(int i) -> void
        {
            p[i] += sigma * q[i];
            out_u[i] += K * p[i];
        });

        // update q
        stiffness.action(p, a); // a = S(p)

        cs = std::cos(omega * n * dt);
        sn = std::sin(omega * n * dt);
        Khalf = C * filter(n + 0.5) / omega;
        forall(ndof, [=] __device__(int i) -> void
        {
            double M = m(i);
            double H = h(i);

            double inv = 1.0 / (M + theta * H);
            double alpha = (M - theta * H) * inv;
            double beta = sigma * inv;

            double acc = -a[i];
            if (d_f)
                acc += cs * b[i] + sn * c[i];

            q[i] = alpha * q[i] + beta * acc;
            out_v[i] += Khalf * q[i];
        });
    }
}
