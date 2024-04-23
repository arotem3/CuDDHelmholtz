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
        const_imat_wrapper global_indices() const
        {
            return reshape(gI.data(), gI.shape()[0], gI.shape()[1]);
        }

        /// @brief returns the sizes of the subspaces. That is, sizes(p) is the
        /// size of subspace p.
        const_ivec_wrapper sizes() const
        {
            return reshape(s_dof.data(), s_dof.size());
        }

        /// @brief returns the elements in each subspace. That is elements(el, p)
        /// is the element index of the el-th element in subspace p. 
        const_imat_wrapper elements() const
        {
            return reshape(elems.data(), elems.shape()[0], elems.shape()[1]);
        }

        /// @brief returns the number of elements in each subspace. That is
        /// n_elems(p) is the number of elements in subspace p. 
        const_ivec_wrapper n_elems() const
        {
            return reshape(s_elems.data(), s_elems.size());
        }

        /// @brief returns the boundary faces of each subspace.
        /// That is faces(f, p) is the face index of the f-th boundary face of
        /// subspace p.
        const_imat_wrapper faces() const
        {
            return reshape(_faces.data(), _faces.shape()[0], _faces.shape()[1]);
        }

        /// @brief returns the number of boundary faces in each subspace. That
        /// is n_faces(p) is the number of faces in subspace p. 
        const_ivec_wrapper n_faces() const
        {
            return reshape(s_faces.data(), s_faces.size());
        }

        /// @brief returns the indices of subspace degrees of freedom
        /// corresponding to the local element degrees of freedom. Namely,
        /// subspace_indices(i,j,el,p) returns the subspace index of the degree
        /// of freedom corresponding to the (i,j) node on element el.
        TensorWrapper<4, const int> subspace_indices() const
        {
            return reshape(sI.data(), sI.shape(0), sI.shape(1), sI.shape(2), sI.shape(3));
        }

        /// @brief returns the indices of the face space degrees of freedom from
        /// face local indices in subspaces's face space.
        ///
        /// That is face_indices(i, f, p) is the face space index of i-th degree of
        /// freedom on face f in subspace p.
        const_icube_wrapper face_indices() const
        {
            return reshape(fI.data(), fI.shape()[0], fI.shape()[1], fI.shape()[2]);
        }

        /// @brief returns the indices of the subspace degrees of freedom from
        /// the face space degrees of freedom.
        ///
        /// That is face_proj(i, p) is the p-th subspace index of the i-th face
        /// space degree of freedom
        const_imat_wrapper face_proj() const
        {
            return reshape(pI.data(), pI.shape()[0], pI.shape()[1]);
        }

        /// @brief returns the number of face spaces degrees of freedom
        /// associated with each space. That is fsizes(p) is the number of
        /// degrees of freedom in the face space of subspace p.
        const_ivec_wrapper fsizes() const
        {
            return reshape(s_fdof.data(), s_fdof.size());
        }

        /// @brief connectivity_map(:, k) = [p, q, i, j] indicating the
        /// subspaces p and q share a face degree of freedom, and that degree of
        /// freedom corresponds to the i-th face DOF of subspace p, and the j-th
        /// face DOF of subspace q. The map is sorted with respect to p, and
        /// does not store the symmetric set [q, p, j, i].
        const_imat_wrapper connectivity_map() const
        {
            return reshape(cmap.data(), 4, cmap.shape()[1]);
        }

    private:
        int n_spaces;

        imat gI;
        ivec s_dof;
        
        imat elems;
        ivec s_elems;
        
        imat _faces;
        ivec s_faces;

        Tensor<4, int> sI;
        icube fI;
        imat pI;
        ivec s_fdof;

        imat cmap;
    };
} // namespace cuddh

#endif
