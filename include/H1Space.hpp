#ifndef CUDDH_H1_SPACE_HPP
#define CUDDH_H1_SPACE_HPP

#include <unordered_set>

#include "Tensor.hpp"
#include "Mesh2D.hpp"
#include "Basis.hpp"
#include "Operator.hpp"

#include "HostDeviceArray.hpp"
#include "forall.hpp"

namespace cuddh
{
    /// @brief abstract representation of H1 finite element space on a mesh of
    /// quad elements with tensor product basis functions. This class defines
    /// the global ordering of degrees of freedom for vectors in this space.
    class H1Space
    {
    public:
        H1Space(const Mesh2D& mesh, const Basis& basis);

        /// @brief returns the dimension of the space, i.e. the number of
        /// degrees of freedom. 
        int size() const
        {
            return ndof;
        }

        /// @brief returns the inidices of the global degrees of freedom from
        /// element local indicies.
        /// The output has shape (n_basis, n_basis, n_elem).
        const_icube_wrapper global_indices(MemorySpace m) const
        {
            return reshape(_I.read(m), n_basis, n_basis, n_elem);
        }

        /// @brief returns a reference to the mesh
        const Mesh2D& mesh() const
        {
            return _mesh;
        }

        /// @brief returns a reference to the basis set
        const Basis& basis() const
        {
            return _basis;
        }

        /// @brief returns the physical coordinates corresponding to collocation
        /// point of each nodal DOF. The output has shape (2, ndof).
        const_dmat_wrapper physical_coordinates(MemorySpace m) const
        {
            return reshape(_xy.read(m), 2, ndof);
        }

    private:
        const int n_elem;
        const int n_basis;
        const Mesh2D& _mesh;
        const Basis& _basis;
        int ndof;

        host_device_ivec _I;
        host_device_dvec _xy;
    };

    /// @brief abstract representation of a subspace of an H1Space spanned by
    /// the basis functions with support on the specified faces
    class FaceSpace
    {
    public:
        FaceSpace(const H1Space& fem, int n_faces, const int * faces);

        /// @brief returns the dimension of the space 
        int size() const
        {
            return ndof;
        }
        
        /// @brief returns the number of faces 
        int n_faces() const
        {
            return _n_faces;
        }

        /// @brief returns the face indices of the faces in the space 
        const_ivec_wrapper faces(MemorySpace m) const
        {
            return reshape(_faces.read(m), _n_faces);
        }

        /// @brief returns the indices of the FaceSpace degrees of freedom
        /// corresponding to the local face indices. Specifically,
        /// subspace_indices(i, f) is the subspace index of the i-th basis
        /// function on face f. These indices range from 0 to this->size()-1.
        const_imat_wrapper subspace_indices(MemorySpace m) const
        {
            return reshape(_I.read(m), n_basis, _n_faces);
        }

        /// @brief returns the indicies of the global degrees of freedom in the
        /// H1Space relative to the FaceSpace. that is, global_indicies(i) is
        /// the index in H1Space corresponding to the i-th FaceSpace degree of
        /// freedom.
        const_ivec_wrapper global_indices(MemorySpace m) const
        {
            return reshape(_proj.read(m), ndof);
        }

        /// @brief project H1Space vector to FaceSpace vector
        /// @param x DEVICE. H1Space vector
        /// @param y DEVICE. FaceSpace vector
        void restrict(const double * x, double * y) const;

        /// @brief Transpose of restrict. Extend FaceSpace vector to H1Space
        /// @param x DEVICE. FaceSpace vector
        /// @param y DEVICE. H1Space vector. On exit, y <- y + P' * x where P is the restriction operator.
        void prolong(const double * x, double * y) const;

        /// @brief returns the global H1Space
        const H1Space& h1_space() const
        {
            return fem;
        }

        /// @brief returns the edge metrics for the faces in the FaceSpace 
        const Mesh2D::EdgeMetricCollection& metrics(const QuadratureRule& quad) const;

    private:
        const H1Space& fem;
        const int _n_faces;
        const int n_basis;
        int ndof;

        host_device_ivec _I;
        host_device_ivec _faces;
        host_device_ivec _proj;

        mutable std::unordered_map<std::string, Mesh2D::EdgeMetricCollection> _metrics;
    };

    /// @brief projects an H1Space vector to H1_0
    class H1_0
    {
    public:
        /// @brief initialize projector
        /// @param fs FaceSpace of all boundary Dirichlet faces
        H1_0(const FaceSpace& fs_) : fs{fs_} {}

        /// @brief projects an H1Space vector onto the H1_0 inplace. 
        void action(double * y) const
        {
            const int ndof = fs.size();
            const int * I = fs.global_indices(MemorySpace::DEVICE);

            forall(ndof, [=] __device__ (int i) -> void
            {
                y[I[i]] = 0.0;
            });
        }

    private:
        const FaceSpace& fs;
    };
} // namespace cuddh


#endif