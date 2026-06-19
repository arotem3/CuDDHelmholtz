#include "DD3D/DDFaceMassMatrix3D.hpp"

using namespace cuddh;

template <typename scalar_t>
static thrust::universal_vector<scalar_t> init_face_mass(const EnsembleSpace3D &efem, const GridFunc3D<double> *a)
{
    const H1Space3D &fem = efem.h1_space();
    const Basis &basis = fem.basis();
    const auto &quad = basis.quadrature();
    const auto &mesh = fem.mesh().to_device();

    const int n_basis = basis.size();
    const int mx_n_faces = efem.max_n_faces();
    const int mx_fdofs = efem.max_fsize();
    const int n_domains = efem.size();

    thrust::universal_vector<scalar_t> m(mx_fdofs * n_domains, scalar_t(0));
    auto H = reshape(m, mx_fdofs, n_domains);

    auto w = quad.w(MemorySpace::DEVICE);
    auto x = quad.x(MemorySpace::DEVICE);

    auto n_faces = efem.n_faces(MemorySpace::DEVICE);
    auto faces = efem.faces(MemorySpace::DEVICE);
    auto sides = efem.face_sides(MemorySpace::DEVICE);
    auto I = efem.face_indices(MemorySpace::DEVICE);

    TensorWrapper<4, const double> A;
    if (a)
        A = a->read(MemorySpace::DEVICE);

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

        if (A)
        {
            const FaceConnectivity connectivity = mesh.face_connectivity(faces(f, p));
            const int side = sides(f, p);
            int ip = i;
            int jp = j;
            if (side == 1)
            {
                const auto [ip1, jp1] = permute_face_index(n_basis, i, j, connectivity.permutation);
                ip = ip1;
                jp = jp1;
            }

            const auto [xv, yv, zv] = face2vol(n_basis, ip, jp, connectivity.label[side]);
            const int el = connectivity.elements[side];
            value *= A(xv, yv, zv, el);
        }

        atomicAdd(&H(idx, p), static_cast<scalar_t>(value));
    });

    return m;
}

template <typename scalar_t>
DDFaceMassMatrix3D<scalar_t>::DDFaceMassMatrix3D(const EnsembleSpace3D &efem)
    : mx_fdofs(efem.max_fsize()), n_domains(efem.size())
{
    m = init_face_mass<scalar_t>(efem, nullptr);
}

template <typename scalar_t>
DDFaceMassMatrix3D<scalar_t>::DDFaceMassMatrix3D(const EnsembleSpace3D &efem, const GridFunc3D<double> &a)
    : mx_fdofs(efem.max_fsize()), n_domains(efem.size())
{
    m = init_face_mass<scalar_t>(efem, &a);
}

namespace cuddh
{
    template class DDFaceMassMatrix3D<float>;
    template class DDFaceMassMatrix3D<double>;
} // namespace cuddh
