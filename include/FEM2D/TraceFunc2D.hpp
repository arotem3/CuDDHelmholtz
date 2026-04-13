#pragma once

#include "FEM2D/GridFunc2D.hpp"
#include "FEM2D/H1Space2D.hpp"
#include "HostDeviceArray.hpp"
#include "Tensor.hpp"
#include "cuddh_config.hpp"

namespace cuddh
{
    template <typename value_t>
    class TraceFunc2D
    {
    public:
        using value_type = value_t;

        explicit TraceFunc2D(const TraceSpace2D &tr) : tr{tr}
        {
            int nb = tr.h1_space().basis().size();
            int nf = tr.n_faces();

            values.resize(nb * nf);
        }

        int n_faces() const { return tr.n_faces(); }
        int n_basis() const { return tr.h1_space().basis().size(); }
        const TraceSpace2D &trace_space() const { return tr; }

        MatrixWrapper<const value_t> read(MemorySpace m) const { return reshape(values.read(m), n_basis(), n_faces()); }
        MatrixWrapper<value_t> write(MemorySpace m) { return reshape(values.write(m), n_basis(), n_faces()); }
        MatrixWrapper<value_t> read_write(MemorySpace m) { return reshape(values.read_write(m), n_basis(), n_faces()); }

    private:
        const TraceSpace2D &tr;
        HostDeviceArray<value_t> values;
    };

    template <typename value_t>
    TraceFunc2D<value_t> trace(const TraceSpace2D &tr, const GridFunc2D<value_t> &gf)
    {
        const int nb = gf.n_basis();

        TraceFunc2D<value_t> tf(tr);
        auto T = tf.write(MemorySpace::DEVICE);

        auto F = gf.read(MemorySpace::DEVICE);

        const auto &mesh = tr.h1_space().mesh().to_device();
        const auto faces = tr.faces(MemorySpace::DEVICE);

        forall_1d(nb, faces.size(), [=] __device__(int f) mutable {
            const auto i = threadIdx.x;

            __shared__ EdgeConnectivity connectivity;
            if (i == 0)
                connectivity = mesh.edge_connectivity(faces[f]);
            __syncthreads();

            const int el = connectivity.elements[0];

            const auto [x, y] = edge2vol(nb, i, connectivity.labels[0]);
            T(i, f) = F(x, y, el);
        });

        return tf;
    }
} // namespace cuddh
