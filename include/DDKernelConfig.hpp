#pragma once

namespace cuddh
{
    /// @brief Selects the subdomain solver used inside DDSubstructuredOperator and DDSubstructuredOperator3D.
    enum class SubdomainSolver
    {
        WaveHoltz,
        MINRES,
#ifdef CUDDH_HAS_CUDSS
        SparseDirect,
#endif
    };

    /* Kernels are deployed with `block_size` threads per block with each thread computing `tdof` DOFs.
     * if `block_size == Default`, then it is determined from the EnsembleSpace and `tdof`.
     * if `tdof <= 0`, then it is determined from the EnsembleSpace and `block_size`.
     * if both are unspecified, then some viable configuration will be selected.
     * In 3D, `block_size` is always determined by the basis size and is ignored.
     */
    struct DDKernelConfig
    {
        enum BlockSize
        {
            Default = 0,
            t256 = 256,
            t512 = 512,
            t1024 = 1024,
        };

        BlockSize block_size = Default;
        int tdof = 0;
    };
} // namespace cuddh
