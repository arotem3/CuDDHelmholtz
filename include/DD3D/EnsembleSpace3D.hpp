#pragma once

#include "FEM3D/H1Space3D.hpp"
#include "LambdaDof.hpp"
#include "Tensor.hpp"

namespace cuddh
{
    class EnsembleSpace3D
    {
    public:
        /**
         * @brief Construct a new EnsembleSpace3D by specifying the global H1Space3D
         * and the association of each element to a subspace.
         *
         * @param fem the global H1Space3D
         * @param n_spaces number of spaces in ensemble
         * @param element_labels has length n_elem. element_labels[el]
         * indicates which subspace element el belongs to.
         */
        EnsembleSpace3D(const H1Space3D &fem, int n_spaces, const int *element_labels);

        /// @brief returns the number of subspaces
        int size() const { return n_spaces; }

        /**
         * @brief returns the global indices of the subspace degrees of
         * freedom. That is, global_indices(i, p) is the global index of the
         * i-th degree of freedom of subspace p.
         */
        const_imat_wrapper global_indices(MemorySpace m) const { return reshape(sp_global.read(m), mx_ndof, n_spaces); }

        /**
         * @brief returns the sizes of the subspaces. That is, sizes(p) is the
         * size of subspace p.
         */
        const_ivec_wrapper sizes(MemorySpace m) const { return reshape(sp_dof.read(m), n_spaces); }

        const_ivec_wrapper fsizes(MemorySpace m) const { return reshape(fp_dof.read(m), n_spaces); }

        /**
         * @brief returns the maximum size of any subspace. That is, the maximum of sizes.
         */
        int max_size() const { return mx_ndof; }

        int max_fsize() const { return mx_fdof; }

        /**
         * @brief returns the elements in each subspace. That is elements(el, p)
         * is the element index of the el-th element in subspace p.
         */
        const_imat_wrapper elements(MemorySpace m) const { return reshape(sp_elems.read(m), mx_elems, n_spaces); }

        /**
         * @brief returns the number of elements in each subspace. That is
         * n_elems(p) is the number of elements in subspace p.
         */
        const_ivec_wrapper n_elems(MemorySpace m) const { return reshape(sp_n_elems.read(m), n_spaces); }

        /**
         * @brief returns the maximum number of elements in any subspace. That is, the maximum of n_elems.
         */
        int max_n_elem() const { return mx_elems; }

        /**
         * @brief returns the boundary faces of each subspace.
         * That is faces(f, p) is the face index of the f-th boundary face of
         * subspace p.
         */
        const_imat_wrapper faces(MemorySpace m) const { return reshape(sp_faces.read(m), mx_faces, n_spaces); }

        /**
         * @brief returns the number of boundary faces in each subspace. That
         * is n_faces(p) is the number of faces in subspace p.
         */
        const_ivec_wrapper n_faces(MemorySpace m) const { return reshape(sp_n_faces.read(m), n_spaces); }

        /**
         * @brief returns the maximum number of faces in any subspace.  That is, the maximum of n_faces.
         */
        int max_n_faces() const { return mx_faces; }

        /**
         * @brief returns the indices of the element space degrees of freedom
         * in each subspace. That is subspace_indices(i, j, k, el, p) is the
         * index of the (i, j, k) degree of freedom on element el in subspace p.
         */
        TensorWrapper<5, const int> subspace_indices(MemorySpace m) const
        {
            return reshape(sp_indices.read(m), n_basis, n_basis, n_basis, mx_elems, n_spaces);
        }

        /**
         * @brief returns the indices of the face space degrees of freedom from
         * face local indices in subspaces's face space.
         *
         * That is face_indices(i, j, f, p) is the face space index of (i,j) degree of
         * freedom on face f in subspace p.
         */
        TensorWrapper<4, const int> face_indices(MemorySpace m) const
        {
            return reshape(fp_indices.read(m), n_basis, n_basis, mx_faces, n_spaces);
        }

        /**
         * @brief returns the connectivity map between the shared degrees of freedom.
         *
         * Each element is a LambdaDof, which stores the two subspace indices, the
         * two local face-DOF indices (one per subspace), and the geometric face
         * mass weight at that quadrature node.
         */
        auto connectivity_map(MemorySpace m) const { return reshape(cmap.read(m), n_shared_dofs); }

    private:
        const int n_spaces;
        const int n_basis;
        int mx_elems;
        int mx_faces;
        int mx_ndof;
        int mx_fdof;
        int n_shared_dofs;

        host_device_ivec sp_global;  // (mx_ndof, n_spaces) global indices of the subspace degrees of freedom
        host_device_ivec sp_dof;     // (n_spaces,) sizes of the subspaces
        host_device_ivec sp_n_elems; // (n_spaces,) number of elements in each subspace
        host_device_ivec sp_elems;   // (mx_elems, n_spaces) elements in each subspace
        host_device_ivec sp_n_faces; // (n_spaces,) number of faces in each subspace
        host_device_ivec sp_faces;   // (mx_faces, n_spaces) faces in each subspace
        host_device_ivec sp_indices; // (n_basis, n_basis, n_basis, mx_elems, n_spaces) indices of the element space
                                     // degrees of freedom
        host_device_ivec
            fp_indices;          // (n_basis, n_basis, mx_faces, n_spaces) indices of the face space degrees of freedom
        host_device_ivec fp_dof; // (n_spaces,) number of trace space degrees of freedom in each subspace
        HostDeviceArray<LambdaDof> cmap; // (n_shared_dofs,) connectivity map
    };

    EnsembleSpace3D partition_uniform_cube(const H1Space3D &fem, dim3 mesh_dims, dim3 block_dims = {4, 4, 2});
} // namespace cuddh
