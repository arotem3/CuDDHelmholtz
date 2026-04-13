#pragma once

#include "DD2D/EnsembleSpace.hpp"
#include "FEM2D/GridFunc2D.hpp"

namespace cuddh
{
    template <typename value_t>
    class DDTraceFunc2D
    {
    public:
        using value_type = value_t;

        explicit DDTraceFunc2D(const EnsembleSpace &efem) : efem{efem}
        {
            int nb = efem.h1_space().basis().size();
            int mxf = efem.max_n_faces();
            int nsp = efem.size();

            values.resize(nb * mxf * nsp);
        }

        int n_basis() const { return efem.h1_space().basis().size(); }
        int max_n_faces() const { return efem.max_n_faces(); }
        int n_subdomains() const { return efem.size(); }
        const EnsembleSpace &ensemble_space() const { return efem; }

        TensorWrapper<3, const value_t> read(MemorySpace m) const
        {
            return reshape(values.read(m), n_basis(), max_n_faces(), n_subdomains());
        }
        TensorWrapper<3, value_t> write(MemorySpace m)
        {
            return reshape(values.write(m), n_basis(), max_n_faces(), n_subdomains());
        }
        TensorWrapper<3, value_t> read_write(MemorySpace m)
        {
            return reshape(values.read_write(m), n_basis(), max_n_faces(), n_subdomains());
        }

    private:
        const EnsembleSpace &efem;
        HostDeviceArray<value_t> values;
    };

    template <typename value_t>
    DDTraceFunc2D<value_t> subdomain_trace(const EnsembleSpace &efem, const GridFunc2D<value_t> &gf)
    {
        const int nb = gf.n_basis();

        DDTraceFunc2D<value_t> tf(efem);
        auto T = tf.write(MemorySpace::DEVICE);

        auto F = gf.read(MemorySpace::DEVICE);

        const auto &mesh = efem.h1_space().mesh().to_device();
        const auto faces = efem.faces(MemorySpace::DEVICE);
        const auto n_faces = efem.n_faces(MemorySpace::DEVICE);
        const auto sides = efem.face_sides(MemorySpace::DEVICE);

        forall_1d(nb, faces.size(), [=] __device__(int b) mutable {
            const auto i = threadIdx.x;
            const int f = b % faces.shape(0);
            const int subsp = b / faces.shape(0);

            if (f >= n_faces[subsp])
                return;

            __shared__ EdgeConnectivity connectivity;
            if (i == 0)
                connectivity = mesh.edge_connectivity(faces(f, subsp));
            __syncthreads();

            const int side = sides(f, subsp);
            const int el = connectivity.elements[side];
            const int j = (side == 1) ? permute_edge_index(nb, i, connectivity.permutation) : i;

            const auto [x, y] = edge2vol(nb, j, connectivity.labels[side]);
            T(i, f, subsp) = F(x, y, el);
        });

        return tf;
    }
} // namespace cuddh
