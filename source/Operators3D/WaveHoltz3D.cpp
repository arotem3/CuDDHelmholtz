#include "Operators3D/WaveHoltz3D.hpp"

using namespace cuddh;

static void init_face_mass(const TraceSpace3D &tr, const double *d_a, double *d_m)
{
    const int n_faces = tr.n_faces();
    const int n_basis = tr.h1_space().basis().size();

    auto& quad = tr.h1_space().basis().quadrature();

    host_device_dvec _w(n_basis);
    double * h_w = _w.host_write();
    for (int i = 0; i < n_basis; ++i)
        h_w[i] = quad.w(i);
    auto w = reshape(_w.device_read(), n_basis);

    host_device_dvec _x(n_basis);
    double * h_x = _x.host_write();
    for (int i = 0; i < n_basis; ++i)
        h_x[i] = quad.x(i);
    auto x = reshape(_x.device_read(), n_basis);

    DeviceMesh3D mesh = tr.h1_space().mesh().to_device();

    auto I = tr.subspace_indices(MemorySpace::DEVICE);
    auto J = tr.global_indices(MemorySpace::DEVICE);

    forall_2d(n_basis, n_basis, n_faces, [=] __device__ (int f) mutable -> void
    {
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

WaveHoltz3D::WaveHoltz3D(double omega, double maxvel, const double * a2x, const double * ax, const H1Space3D& fem_, const TraceSpace3D& fs_)
    : omega(omega),
      ndof(fem_.size()),
      fem(fem_),
      fs(fs_),
      stiffness(fem_),
      M(a2x, fem_),
      H(ndof),
      acc(ndof),
      acc1(ndof),
      w(2*ndof)
{
    init_face_mass(fs, ax, H.device_write());

    double T = 2.0 * M_PI / omega;
    double p = fem.basis().size();
    dt = 2.0 * fem.mesh().h() / (p * p * maxvel); // CFL condition
    dt = std::min(dt, T / 4.0); // at least 4 points per wavelength

    nt = std::ceil(T / dt);
    dt = T / nt;

    double tan = std::tan(0.5 * omega * dt);
    theta = tan / (0.5 * omega);
    sigma = std::cos(0.5 * omega * dt); sigma *= sigma;
    shift = 0.25 - 0.25 * tan * tan;
}

void WaveHoltz3D::action(double c, const double * x, double * y) const
{
    axpby(2*ndof, c, x, 1.0, y); // y <- y + c * x
    S(-c, x, y); // y <- y - c * S(x) = y + c * (I - S) * x
}

void WaveHoltz3D::action(const double * x, double * y) const
{
    copy(2*ndof, x, y); // y <- x
    S(-1.0, x, y); // y <- y - S(x) = y + (I - S) * x
}

void WaveHoltz3D::evolve_project(double C, const double * d_u, const double * d_f, double * d_out) const
{
    if (not d_u && not d_f)
        return;

    if (not d_out)
        throw std::runtime_error("WaveHoltz::evolve_project: d_out is null");

    const int ndof = this->ndof;
    const double omega = this->omega;
    const double da = 0.5 * theta * theta;
    const double half_theta = 0.5 * theta;
    const double theta = this->theta;
    const double sigma = this->sigma;

    double * d_w = w.device_write();
    double * p = d_w;
    double * q = d_w + ndof;

    if (d_u)
    {
        forall(ndof, [=] __device__ (int i) -> void
        {
            p[i] = d_u[i];
            q[i] = omega * d_u[i + ndof];
        });
    }
    else
        zeros(2*ndof, d_w);

    const double * b = (d_f) ? d_f : nullptr;
    const double * c = (d_f) ? d_f + ndof : nullptr;

    auto m = diagonal_mass(M, MemorySpace::DEVICE);
    auto h = reshape(H.device_read(), ndof);

    double * a = acc.device_write();
    double * a1 = acc1.device_write();

    if (d_u)
    {
        stiffness.action(p, a); // a = -S(u)
        forall(ndof, [=] __device__ (int i) -> void
        {
            a[i] = -a[i] - h[i] * q[i]; // a = a - H * q
        });
    }
    else
    {
        zeros(ndof, a); // a(0) = 0
    }

    if (d_u)
    {
        double weight = C * filter(0);
        forall(ndof, [=] __device__ (int i) -> void
        {
            d_out[i] += weight * p[i];
            d_out[i + ndof] += weight * q[i] / omega;
        });
    }

    double t = 0;
    for (int it = 1; it < nt; ++it)
    {
        double cs = 0.5 * std::cos(omega * t) + 0.5 * std::cos(omega * (t + dt));
        double sn = 0.5 * std::sin(omega * t) + 0.5 * std::sin(omega * (t + dt));

        // p1 <- p + theta * q + 0.5 * theta^2 * M \ (sigma * a + cs * b + sn * c)
        forall(ndof, [=] __device__ (int i) -> void
        {
            double F = sigma * a[i];
            if (d_f)
                F += cs * b[i] + sn * c[i];

            p[i] += theta * q[i] + da * F / m[i];
        });

        // a1 <- S(p)
        stiffness.action(p, a1);

        // q1 <- (M + theta / 2 * H) \ (M * q + theta / 2 * a - theta / 2 * a1 + 2 * cs * b + 2 * sn * c)
        // a1 <- -a1 - H * q1
        forall(ndof, [=] __device__ (int i) -> void
        {
            double F = 0.5 * a[i] - 0.5 * a1[i];
            
            if (d_f)
                F += cs * b[i] + sn * c[i];

            double r = m[i] * q[i] + theta * F;

            q[i] = r / (m[i] + half_theta * h[i]);
            a[i] = -a1[i] - h[i] * q[i];
        });

        t += dt;

        // axpby(2*ndof, C * filter(t), d_w, 1.0, d_out);
        double weight = C * filter(t);
        forall(ndof, [=] __device__ (int i) -> void
        {
            d_out[i] += weight * p[i];
            d_out[i + ndof] += weight * q[i] / omega;
        });
    }
}
