#include "DD3D/DDFaceMassMatrix3D.hpp"

using namespace cuddh;

static void init_face_mass(const TraceSpace3D &tr, const EnsembleSpace3D &efem, MatrixWrapper<float> H)
{
    const Basis &basis = tr.h1_space().basis();
    const auto &quad = basis.quadrature();
    const auto &mesh = tr.h1_space().mesh().to_device();

    const int n_basis = basis.size();
    const int mx_n_faces = efem.max_n_faces();
    const int n_domains = efem.size();

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

    auto n_faces = efem.n_faces(MemorySpace::DEVICE);
    auto faces = efem.faces(MemorySpace::DEVICE);
    auto I = efem.face_indices(MemorySpace::DEVICE);
    
    forall_2d(n_basis, n_basis, mx_n_faces * n_domains, [=] __device__ (int b) mutable -> void
    {
        const int i = threadIdx.x;
        const int j = threadIdx.y;
        const int f = b % mx_n_faces;
        const int p = b / mx_n_faces;

        if (f > n_faces(p))
            return;

        __shared__ QuadFace face;
        if (i == 0 && j == 0)
            face = mesh.face(f);
        
        const int idx = I(i, j, f, p);
        double value = w(i) * w(j);

        const double2 r{ x(i), x(j) };

        __syncthreads();

        value *= face.measure(r);
        atomicAdd(&H(idx, p), static_cast<float>(value));
    });
}

DDFaceMassMatrix3D::DDFaceMassMatrix3D(const TraceSpace3D &tr, const EnsembleSpace3D &efem)
    : mx_fdofs(efem.max_fdof()),
      n_domains(efem.size()),
      m(mx_fdofs * n_domains)
{
    init_face_mass(tr, efem, reshape(m.device_write(), mx_fdofs, n_domains));
}
