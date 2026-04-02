#include "DD3D/DDFaceMassMatrix3D.hpp"

using namespace cuddh;

static thrust::universal_vector<float> init_face_mass(const H1Space3D &fem, const EnsembleSpace3D &efem)
{
    const Basis &basis = fem.basis();
    const auto &quad = basis.quadrature();
    const auto &mesh = fem.mesh().to_device();

    const int n_basis = basis.size();
    const int mx_n_faces = efem.max_n_faces();
    const int mx_fdofs = efem.max_fsize();
    const int n_domains = efem.size();

    thrust::universal_vector<float> m(mx_fdofs * n_domains, 0.0f);
    auto H = reshape(m, mx_fdofs, n_domains);

    thrust::universal_vector<float> u_w(n_basis);
    for (int i = 0; i < n_basis; ++i)
        u_w[i] = quad.w(i);
    auto w = reshape(u_w, n_basis);

    thrust::universal_vector<float> u_x(n_basis);
    for (int i = 0; i < n_basis; ++i)
        u_x[i] = quad.x(i);
    auto x = reshape(u_x, n_basis);

    auto n_faces = efem.n_faces(MemorySpace::DEVICE);
    auto faces = efem.faces(MemorySpace::DEVICE);
    auto I = efem.face_indices(MemorySpace::DEVICE);

    forall_2d(n_basis, n_basis, mx_n_faces * n_domains, [=] __device__(int b) mutable -> void {
        const int i = threadIdx.x;
        const int j = threadIdx.y;
        const int f = b % mx_n_faces;
        const int p = b / mx_n_faces;

        if (f >= n_faces(p))
            return;

        __shared__ QuadFace face;
        if (i == 0 && j == 0)
            face = mesh.face(faces(f, p));

        const int idx = I(i, j, f, p);
        double value = w(i) * w(j);

        const double2 r{x(i), x(j)};

        __syncthreads();

        value *= face.measure(r);
        atomicAdd(&H(idx, p), static_cast<float>(value));
    });

    return m;
}

DDFaceMassMatrix3D::DDFaceMassMatrix3D(const H1Space3D &fem, const EnsembleSpace3D &efem)
    : mx_fdofs(efem.max_fsize()), n_domains(efem.size())
{
    m = init_face_mass(fem, efem);
}
