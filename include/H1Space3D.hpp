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
        /// point of each nodal DOF. The output has shape (ndof,).
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

    class TraceSpace3D
    {
    public:
        /// @brief Construct a TraceSpace3D from an H1Space3D and a list of faces
        TraceSpace3D(const H1Space3D &fem, int n_faces, const int *faces);

        /// @brief returns the dimension of the space
        int size() const
        {
            return ndof;
        }

        /// @brief returns the number of faces in the space
        int n_faces() const
        {
            return nf;
        }

        /// @brief returns the face indices of the faces in the space
        const_ivec_wrapper faces(MemorySpace m) const
        {
            return reshape(_faces.read(m), nf);
        }

        /// @brief returns the indices of the TraceSpace3D degrees of freedom
        /// corresponding to the local face indices. Specifically,
        /// subspace_indices(i, j, f) is the subspace index of the (i,j) basis
        /// function on face f. These indices range from 0 to this->size()-1.
        const_icube_wrapper subspace_indices(MemorySpace m) const
        {
            return reshape(_I.read(m), n_basis, n_basis, nf);
        }

        /// @brief returns the indicies of the global degrees of freedom in the
        /// H1Space3D relative to the TraceSpace3D. that is, global_indicies(i) is
        /// the index in H1Space3D corresponding to the i-th TraceSpace3D degree of
        /// freedom.
        const_ivec_wrapper global_indices(MemorySpace m) const
        {
            return reshape(_proj.read(m), ndof);
        }

        /// @brief project H1Space3D vector to TraceSpace3D vector
        /// @param x DEVICE. H1Space3D vector
        /// @param y DEVICE. TraceSpace3D vector
        void restrict(const double * x, double * y) const;

        /// @brief Transpose of restrict. Extend TraceSpace3D vector to H1Space3D
        /// @param x DEVICE. TraceSpace3D vector
        /// @param y DEVICE. H1Space3D vector. On exit, y <- y + P' * x where P is the restriction operator.
        void prolong(const double * x, double * y) const;

        /// @brief Project H1Space3D space vector to orthogonal complement of
        /// TraceSpace3D. I.e. set face values to zero.
        /// @param x DEVICE. H1Space3D vector. On exit, x <- x - P' * P * x where
        /// P is the restriction operator and P' is prolongation operator.
        void orth(double * x) const;

        /// @brief returns the global H1Space3D
        const H1Space3D& h1_space() const
        {
            return fem;
        }

    private:
        const H1Space3D &fem;
        const int nf;
        const int n_basis;
        int ndof;

        host_device_ivec _I; // subspace indices (n_basis, n_basis, n_basis, n_faces)
        host_device_ivec _faces; // face indices (n_faces)
        host_device_ivec _proj; // global indices (ndof)
    };
} // namespace cuddh

#endif
