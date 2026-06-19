#pragma once

namespace cuddh
{
    /// @brief Kernel-side descriptor for a substructured (lambda) DOF associated with a face DOF.
    /// Stores the lambda index, its dual index, and the trace operator coefficient.
    template <typename scalar_t>
    struct LambdaDOFData
    {
        int i = -1;       ///< lambda index (negative if unused)
        int j = -1;       ///< dual lambda index (negative if unused)
        scalar_t trOp = 0; ///< trace operator coefficient
    };
} // namespace cuddh
