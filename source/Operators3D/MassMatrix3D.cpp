#include "Operators3D/MassMatrix3D.hpp"

#include "forall.hpp"
#include "linalg.hpp"

using namespace cuddh;

static void init_mass(const H1Space3D &fem, const double *d_a, double *d_M)
{
    const auto &quad = fem.basis().quadrature();
    const auto &mesh = fem.mesh().to_device();

    const int n_elem = fem.mesh().n_elem();
    const int n_basis = quad.size();
    const int ndof = fem.size();

    auto I = reshape(fem.global_indices(MemorySpace::DEVICE), n_basis, n_basis, n_basis, n_elem);

    auto w = quad.w(MemorySpace::DEVICE);
    auto x = quad.x(MemorySpace::DEVICE);

    forall_3d(n_basis, n_basis, n_basis, n_elem, [=] __device__(int el) mutable -> void {
        const int i = threadIdx.x;
        const int j = threadIdx.y;
        const int k = threadIdx.z;

        const int idx = I(i, j, k, el);

        __shared__ HexElement elem;

        if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0)
            elem = mesh.element(el);
        __syncthreads();

        const double3 r{x(i), x(j), x(k)};
        const double detJ = elem.measure(r);

        double m = w(i) * w(j) * w(k) * detJ;
        m *= (d_a) ? d_a[idx] : 1.0;

        atomicAdd(d_M + idx, m);
    });
}

MassMatrix3D::MassMatrix3D(const H1Space3D &fem) : Operator<double>(fem.size()), fem{fem}, _m(fem.size())
{
    init_mass(fem, nullptr, _m.device_write());
}

MassMatrix3D::MassMatrix3D(const double *d_a, const H1Space3D &fem)
    : Operator<double>(fem.size()), fem{fem}, _m(fem.size())
{
    init_mass(fem, d_a, _m.device_write());
}

void MassMatrix3D::action(double c, const double *x, double *y) const
{
    const int n = fem.size();
    auto m = _m.device_read();

    forall(n, [=] __device__(int i) -> void { y[i] += c * m[i] * x[i]; });
}

void MassMatrix3D::action(const double *x, double *y) const
{
    const int n = fem.size();
    auto m = _m.device_read();

    forall(n, [=] __device__(int i) -> void { y[i] = m[i] * x[i]; });
}

InvMassMatrix3D MassMatrix3D::inv() const
{
    return InvMassMatrix3D(*this);
}

static void inv_mass(int n, const double *__restrict__ d_m, double *__restrict__ d_mi)
{
    forall(n, [=] __device__(int i) -> void { d_mi[i] = 1.0 / d_m[i]; });
}

// inplace
static void inv_mass(int n, double *d_m)
{
    forall(n, [=] __device__(int i) -> void { d_m[i] = 1.0 / d_m[i]; });
}

InvMassMatrix3D::InvMassMatrix3D(const H1Space3D &fem) : Operator<double>(fem.size()), fem{fem}, _mi(fem.size())
{
    init_mass(fem, nullptr, _mi.device_write());
    inv_mass(fem.size(), _mi.device_write());
}

InvMassMatrix3D::InvMassMatrix3D(const double *d_a, const H1Space3D &fem)
    : Operator<double>(fem.size()), fem{fem}, _mi(fem.size())
{
    init_mass(fem, d_a, _mi.device_write());
    inv_mass(fem.size(), _mi.device_write());
}

InvMassMatrix3D::InvMassMatrix3D(const MassMatrix3D &M) : Operator<double>(M.fem.size()), fem{M.fem}, _mi(fem.size())
{
    const double *d_m = M._m.device_read();
    double *d_mi = _mi.device_write();

    inv_mass(fem.size(), d_m, d_mi);
}

void InvMassMatrix3D::action(double c, const double *x, double *y) const
{
    const int n = fem.size();
    auto mi = _mi.device_read();

    forall(n, [=] __device__(int i) -> void { y[i] += c * mi[i] * x[i]; });
}

void InvMassMatrix3D::action(const double *x, double *y) const
{
    const int n = fem.size();
    auto mi = _mi.device_read();

    forall(n, [=] __device__(int i) -> void { y[i] = mi[i] * x[i]; });
}

template <int SZ, int NR, typename Lambda>
__global__ static void sum_reduction_kernel(int n, const double *__restrict__ m, const double *x, const double *y,
                                            double *__restrict__ result, Lambda op)
{
    __shared__ double s[SZ];

    const int thread_id = threadIdx.x;
    const int block_id = blockIdx.x;

    double sum = 0.0;

#pragma unroll
    for (int j = 0; j < NR; ++j)
    {
        const int k = thread_id + SZ * (j + NR * block_id);
        if (k < n)
            sum += op(m[k], x[k], y[k]);
    }

    s[thread_id] = sum;

    // tree reduction
    for (int m = SZ >> 1; m > 0; m >>= 1)
    {
        __syncthreads();

        if (thread_id < m)
        {
            s[thread_id] += s[thread_id + m];
        }
    }

    if (thread_id == 0)
    {
        sum = s[0];
        atomicAdd(result, sum);
    }
}

double cuddh::l2_dot(const MassMatrix3D &M, const double *x, const double *y)
{
    const int n = M.fem.size();
    auto m = M._m.device_read();

    host_device_dvec result(1);
    double *d_result = result.device_write();
    dla::zeros(1, d_result);

    auto op = [] __device__(double m, double a, double b) -> double {
        return a * m * b;
    };

    constexpr int block_size = 32;
    constexpr int num_reads = 8;
    constexpr int data_per_block = block_size * num_reads;

    const int n_blocks = (n + data_per_block - 1) / data_per_block;

    sum_reduction_kernel<block_size, num_reads><<<n_blocks, block_size>>>(n, m, x, y, d_result, op);

    return *result.host_read();
}

double cuddh::l2_dist(const MassMatrix3D &M, const double *x, const double *y)
{
    const int n = M.fem.size();
    auto m = M._m.device_read();

    host_device_dvec result(1);
    double *d_result = result.device_write();
    dla::zeros(1, d_result);

    auto op = [] __device__(double m, double a, double b) -> double {
        double e = a - b;
        return e * m * e;
    };

    constexpr int block_size = 32;
    constexpr int num_reads = 8;
    constexpr int data_per_block = block_size * num_reads;

    const int n_blocks = (n + data_per_block - 1) / data_per_block;

    sum_reduction_kernel<block_size, num_reads><<<n_blocks, block_size>>>(n, m, x, y, d_result, op);

    return std::sqrt(*result.host_read());
}
