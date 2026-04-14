#pragma once

#include "FEM3D/GridFunc3D.hpp"
#include "FEM3D/H1Space3D.hpp"
#include "Mesh3D/Connectivity.hpp"
#include "Mesh3D/Mesh3D.hpp"
#include "Tensor.hpp"
#include "cuddh_config.hpp"
#include "forall.hpp"

namespace cuddh
{
    template <typename value_t>
    class TraceFunc3D
    {
    public:
        using value_type = value_t;

        explicit TraceFunc3D(const TraceSpace3D &tr) : _tr{tr}
        {
            int nb = tr.h1_space().basis().size();
            int nf = tr.n_faces();
            values.resize(nb * nb * nf);
        }

        int n_faces() const { return _tr.n_faces(); }
        int n_basis() const { return _tr.h1_space().basis().size(); }
        const TraceSpace3D &trace_space() const { return _tr; }

        TensorWrapper<3, const value_t> read(MemorySpace m) const
        {
            return reshape(values.read(m), n_basis(), n_basis(), n_faces());
        }

        TensorWrapper<3, value_t> write(MemorySpace m)
        {
            return reshape(values.write(m), n_basis(), n_basis(), n_faces());
        }

        TensorWrapper<3, value_t> read_write(MemorySpace m)
        {
            return reshape(values.read_write(m), n_basis(), n_basis(), n_faces());
        }

    private:
        const TraceSpace3D &_tr;
        HostDeviceArray<value_t> values;
    };

    /// @brief Extract values of gf onto boundary faces, returning a TraceFunc3D.
    /// tf(i, j, f) = gf(x, y, z, el) where (x, y, z) are the volume basis indices
    /// corresponding to face basis indices (i, j) on face f.
    template <typename value_t>
    TraceFunc3D<value_t> trace(const TraceSpace3D &tr, const GridFunc3D<value_t> &gf)
    {
        const int nb = gf.n_basis();
        const int n_faces = tr.n_faces();

        const auto d_mesh = tr.h1_space().mesh().to_device();
        const auto faces_d = tr.faces(MemorySpace::DEVICE);

        TraceFunc3D<value_t> tf(tr);
        auto T = tf.write(MemorySpace::DEVICE);
        auto F = gf.read(MemorySpace::DEVICE);

        forall_2d(nb, nb, n_faces, [=] __device__(int f) mutable {
            const int i = threadIdx.x;
            const int j = threadIdx.y;
            const FaceConnectivity fc = d_mesh.face_connectivity(faces_d[f]);
            const auto [x, y, z] = face2vol(nb, i, j, fc.label[0]);
            T(i, j, f) = F(x, y, z, fc.elements[0]);
        });

        return tf;
    }
} // namespace cuddh
