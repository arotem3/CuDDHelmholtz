#pragma once

#include <unordered_set>

#include "Basis.hpp"
#include "HostDeviceArray.hpp"
#include "Mesh2D/Mesh2D.hpp"
#include "Operator.hpp"
#include "Tensor.hpp"
#include "forall.hpp"

namespace cuddh
{
    /// @brief abstract representation of H1 finite element space on a mesh of
    /// quad elements with tensor product basis functions. This class defines
    /// the global ordering of degrees of freedom for vectors in this space.
    class H1Space2D
    {
    public:
        H1Space2D(const Mesh2D &mesh, const Basis &basis);

        /// @brief returns the dimension of the space, i.e. the number of
        /// degrees of freedom.
        int size() const { return ndof; }

        /// @brief returns the inidices of the global degrees of freedom from
        /// element local indicies.
        /// The output has shape (n_basis, n_basis, n_elem).
        const_icube_wrapper global_indices(MemorySpace m) const
        {
            return reshape(_I.read(m), n_basis, n_basis, n_elem);
        }

        /// @brief returns a reference to the mesh
        const Mesh2D &mesh() const { return _mesh; }

        /// @brief returns a reference to the basis set
        const Basis &basis() const { return _basis; }

        /// @brief returns the physical coordinates corresponding to collocation
        /// point of each nodal DOF. The output has shape (2, ndof).
        VectorWrapper<const double2> physical_coordinates(MemorySpace m) const { return reshape(_xy.read(m), ndof); }

    private:
        const int n_elem;
        const int n_basis;
        const Mesh2D &_mesh;
        const Basis &_basis;
        int ndof;

        host_device_ivec _I;
        HostDeviceArray<double2> _xy;
    };

    /// @brief abstract representation of a subspace of an H1Space2D spanned by
    /// the basis functions with support on the specified faces
    class TraceSpace2D
    {
    public:
        TraceSpace2D(const H1Space2D &fem, int n_faces, const int *faces);

        /// @brief returns the dimension of the space
        int size() const { return ndof; }

        /// @brief returns the number of faces
        int n_faces() const { return _n_faces; }

        /// @brief returns the face indices of the faces in the space
        const_ivec_wrapper faces(MemorySpace m) const { return reshape(_faces.read(m), _n_faces); }

        /// @brief returns the indices of the TraceSpace2D degrees of freedom
        /// corresponding to the local face indices. Specifically,
        /// subspace_indices(i, f) is the subspace index of the i-th basis
        /// function on face f. These indices range from 0 to this->size()-1.
        const_imat_wrapper subspace_indices(MemorySpace m) const { return reshape(_I.read(m), n_basis, _n_faces); }

        /// @brief returns the indicies of the global degrees of freedom in the
        /// H1Space2D relative to the TraceSpace2D. that is, global_indicies(i) is
        /// the index in H1Space2D corresponding to the i-th TraceSpace2D degree of
        /// freedom.
        const_ivec_wrapper global_indices(MemorySpace m) const { return reshape(_proj.read(m), ndof); }

        /// @brief project H1Space2D vector to TraceSpace2D vector
        /// @param x DEVICE. H1Space2D vector
        /// @param y DEVICE. TraceSpace2D vector
        void restrict(const double *x, double *y) const;

        /// @brief Transpose of restrict. Extend TraceSpace2D vector to H1Space2D
        /// @param x DEVICE. TraceSpace2D vector
        /// @param y DEVICE. H1Space2D vector. On exit, y <- y + P' * x where P is the restriction operator.
        void prolong(const double *x, double *y) const;

        /// @brief Project H1Space2D space vector to orthogonal complement of
        /// TraceSpace2D. I.e. set face values to zero.
        /// @param x DEVICE. H1Space2D vector. On exit, x <- x - P' * P * x where
        /// P is the restriction operator and P' is prolongation operator.
        void orth(double *x) const;

        /// @brief returns the global H1Space2D
        const H1Space2D &h1_space() const { return fem; }

    private:
        const H1Space2D &fem;
        const int _n_faces;
        const int n_basis;
        int ndof;

        host_device_ivec _I;
        host_device_ivec _faces;
        host_device_ivec _proj;
    };
} // namespace cuddh
