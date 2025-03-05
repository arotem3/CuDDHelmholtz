#ifndef CUDDH_H1_SPACE_3D_HPP
#define CUDDH_H1_SPACE_3D_HPP

#include "cuddh_config.hpp"

#include "Tensor.hpp"
#include "Mesh3D/Mesh3D.hpp"
#include "Basis.hpp"

#include "HostDeviceArray.hpp"
#include "forall.hpp"

namespace cuddh
{
    class H1Space3D
    {
    public:
        H1Space3D(const Mesh3D &mesh, const Basis &basis);

        /// @brief returns the dimension of the space, i.e. the number of
        /// degrees of freedom.
        int size() const
        {
            return ndof;
        }

        /// @brief returns the inidices of the global degrees of freedom from
        /// element local indicies.
        /// The output has shape (n_basis, n_basis, n_basis, n_elem).
        TensorWrapper<4, const int> global_indices(MemorySpace m) const
        {
            return reshape(_I.read(m), n_basis, n_basis, n_basis, n_elem);
        }

        /// @brief returns a reference to the mesh
        const Mesh3D &mesh() const
        {
            return _mesh;
        }

        /// @brief returns a reference to the basis set
        const Basis &basis() const
        {
            return _basis;
        }

        /// @brief returns the physical coordinates corresponding to collocation
        /// point of each nodal DOF. The output has shape (3, ndof).
        VectorWrapper<const double3> physical_coordinates(MemorySpace m) const
        {
            return reshape(_xyz.read(m), ndof);
        }

    private:
        const int n_elem;
        const int n_basis;
        int ndof;

        const Mesh3D &_mesh;
        const Basis &_basis;

        host_device_ivec _I;
        HostDeviceArray<double3> _xyz;
    };
} // namespace cuddh

#endif
