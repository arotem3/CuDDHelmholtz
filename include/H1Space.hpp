#ifndef CUDDH_H1_SPACE_HPP
#define CUDDH_H1_SPACE_HPP

#include <unordered_set>

#include "Tensor.hpp"
#include "Mesh2D.hpp"
#include "Basis.hpp"

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
        const_icube_wrapper global_indices() const
        {
            return const_icube_wrapper(I.data(), n_basis, n_basis, n_elem);
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

    private:
        const int n_elem;
        const int n_basis;
        const Mesh2D& _mesh;
        const Basis& _basis;
        icube I;
        int ndof;
    };
} // namespace cuddh


#endif