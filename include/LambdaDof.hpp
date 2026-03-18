#pragma once

namespace cuddh
{
    /// @brief Descriptor for a pair of degrees of freedom shared between two
    /// subspaces on an interface. Used by EnsembleSpace and EnsembleSpace3D
    /// connectivity maps.
    struct LambdaDof
    {
        int subspaces[2];
        int local_dof_indices[2];
        double face_mass;
    };
} // namespace cuddh
