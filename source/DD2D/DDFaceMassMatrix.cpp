#include "DD2D/DDFaceMassMatrix.hpp"

#include "DD2D/DDTraceFunc2D.hpp"
#include "forall.hpp"

using namespace cuddh;

template <typename scalar_t>
static HostDeviceArray<scalar_t> init_mass(const EnsembleSpace &efem, const DDTraceFunc2D<double> *_a)
{
    const int mx_fdof = efem.max_fsize();
    const int n_domains = efem.size();

    HostDeviceArray<scalar_t> _m(mx_fdof * n_domains);
    auto m = reshape(_m.device_write(), mx_fdof, n_domains);

    auto mesh = efem.h1_space().mesh().to_device();
    auto w = efem.h1_space().basis().quadrature().w(MemorySpace::DEVICE);

    const int nb = w.size();

    auto n_faces = efem.n_faces(MemorySpace::DEVICE);
    auto faces = efem.faces(MemorySpace::DEVICE);
    auto f_inds = efem.face_indices(MemorySpace::DEVICE);

    TensorWrapper<3, const double> a;
    if (_a)
        a = _a->read(MemorySpace::DEVICE);

    forall_1d(nb, mx_fdof * n_domains, [=] __device__(int b) mutable {
        const int i = threadIdx.x;
        const int f = b % mx_fdof;
        const int subsp = b / mx_fdof;

        if (f >= n_faces(subsp))
            return;

        __shared__ Edge edge;
        if (i == 0)
            edge = mesh.edge(faces(f, subsp));
        __syncthreads();

        const double ds = edge.measure();

        const int l = f_inds(i, f, subsp);
        scalar_t ml = ds * w(i);

        if (a)
            ml *= a(i, f, subsp);

        atomicAdd(&m(l, subsp), ml);
    });

    return _m;
}

template <typename scalar_t>
DDFaceMassMatrix<scalar_t>::DDFaceMassMatrix(const EnsembleSpace &efem)
    : mx_fdof{efem.max_fsize()}, n_domains{efem.size()}
{
    m = init_mass<scalar_t>(efem, nullptr);
}

template <typename scalar_t>
DDFaceMassMatrix<scalar_t>::DDFaceMassMatrix(const EnsembleSpace &efem, const GridFunc2D<double> &a)
    : mx_fdof{efem.max_fsize()}, n_domains{efem.size()}
{
    DDTraceFunc2D<double> tr_a = subdomain_trace(efem, a);
    m = init_mass<scalar_t>(efem, &tr_a);
}

namespace cuddh
{
    template class DDFaceMassMatrix<float>;
    template class DDFaceMassMatrix<double>;
} // namespace cuddh
