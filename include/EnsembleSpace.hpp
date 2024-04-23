#ifndef CUDDH_ENSEMBLE_SPACE_HPP
#define CUDDH_ENSEMBLE_SPACE_HPP

#include <array>
#include <utility>
#include <algorithm>

#include "H1Space.hpp"

namespace cuddh
{
    class EnsembleSpace
    {
    public:
        /// @brief initialize an EnsembleSpace by specifying the global H1Space
        /// and the association of each element to a subspace.
        /// @param fem the global H1Space
        /// @param n_spaces number of spaces in ensemble
        /// @param element_labels has length n_elem. element_labels[el]
        /// indicates which subspace element el belongs to.
        EnsembleSpace(const H1Space& fem, int n_spaces, const int * element_labels);

        /// @brief returns the number of subspaces 
        int size() const
        {
            return n_spaces;
        }

        /// @brief returns the global indices of the subspace degrees of
        /// freedom. That is, global_indices(i, p) is the global index of the
        /// i-th degree of freedom of subspace p.
        const_imat_wrapper global_indices(MemorySpace m) const
        {
            return reshape(gI.read(m), mx_ndof, n_spaces);
        }

        /// @brief returns the sizes of the subspaces. That is, sizes(p) is the
        /// size of subspace p.
        const_ivec_wrapper sizes(MemorySpace m) const
        {
            return reshape(s_dof.read(m), n_spaces);
        }

        /// @brief returns the elements in each subspace. That is elements(el, p)
        /// is the element index of the el-th element in subspace p. 
        const_imat_wrapper elements(MemorySpace m) const
        {
            return reshape(elems.read(m), mx_elems, n_spaces);
        }

        /// @brief returns the number of elements in each subspace. That is
        /// n_elems(p) is the number of elements in subspace p. 
        const_ivec_wrapper n_elems(MemorySpace m) const
        {
            return reshape(s_elems.read(m), n_spaces);
        }

        /// @brief returns the boundary faces of each subspace.
        /// That is faces(f, p) is the face index of the f-th boundary face of
        /// subspace p.
        const_imat_wrapper faces(MemorySpace m) const
        {
            return reshape(_faces.read(m), mx_faces, n_spaces);
        }

        /// @brief returns the number of boundary faces in each subspace. That
        /// is n_faces(p) is the number of faces in subspace p. 
        const_ivec_wrapper n_faces(MemorySpace m) const
        {
            return reshape(s_faces.read(m), n_spaces);
        }

        /// @brief returns the indices of subspace degrees of freedom
        /// corresponding to the local element degrees of freedom. Namely,
        /// subspace_indices(i,j,el,p) returns the subspace index of the degree
        /// of freedom corresponding to the (i,j) node on element el.
        TensorWrapper<4, const int> subspace_indices(MemorySpace m) const
        {
            return reshape(sI.read(m), n_basis, n_basis, mx_elems, n_spaces);
        }

        /// @brief returns the indices of the face space degrees of freedom from
        /// face local indices in subspaces's face space.
        ///
        /// That is face_indices(i, f, p) is the face space index of i-th degree of
        /// freedom on face f in subspace p.
        const_icube_wrapper face_indices(MemorySpace m) const
        {
            return reshape(fI.read(m), n_basis, mx_faces, n_spaces);
        }

        /// @brief returns the indices of the subspace degrees of freedom from
        /// the face space degrees of freedom.
        ///
        /// That is face_proj(i, p) is the p-th subspace index of the i-th face
        /// space degree of freedom
        const_imat_wrapper face_proj(MemorySpace m) const
        {
            return reshape(pI.read(m), mx_fdof, n_spaces);
        }

        /// @brief returns the number of face spaces degrees of freedom
        /// associated with each space. That is fsizes(p) is the number of
        /// degrees of freedom in the face space of subspace p.
        const_ivec_wrapper fsizes(MemorySpace m) const
        {
            return reshape(s_fdof.read(m), n_spaces);
        }

        /// @brief connectivity_map(:, k) = [p, q, i, j] indicating the
        /// subspaces p and q share a face degree of freedom, and that degree of
        /// freedom corresponds to the i-th face DOF of subspace p, and the j-th
        /// face DOF of subspace q. The map is sorted with respect to p, and
        /// does not store the symmetric set [q, p, j, i].
        const_imat_wrapper connectivity_map(MemorySpace m) const
        {
            return reshape(cmap.read(m), 4, n_shared_dofs);
        }

    private:
        int n_spaces;
        int n_basis;
        int mx_elems;
        int mx_faces;
        int mx_ndof;
        int mx_fdof;
        int n_shared_dofs;

        host_device_ivec gI;
        host_device_ivec s_dof;
        host_device_ivec elems;
        host_device_ivec s_elems;
        host_device_ivec _faces;
        host_device_ivec s_faces;
        host_device_ivec sI;
        host_device_ivec fI;
        host_device_ivec pI;
        host_device_ivec s_fdof;
        host_device_ivec cmap;
    };
} // namespace cuddh

#endif
