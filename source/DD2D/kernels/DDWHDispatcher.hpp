#pragma once

#include "source/DD2D/DDHKernelImpl.hpp"

#define DECLARE_WH_KERNEL(qual, scalar_t, NB, TDOF, BLOCK_SIZE)                                                       \
    qual template void WHKernelDispatcher<scalar_t>::dispatch_kernel<NB, TDOF, BLOCK_SIZE>(                           \
        const EnsembleSpace &, int, int, const LambdaDOFData<scalar_t> *, const DDStiffnessMatrix<scalar_t> &,        \
        const scalar_t *, const DDWaveHoltz<scalar_t> &, int, const double *, double *, const scalar_t *, scalar_t *, \
        scalar_t *);

#define DECLARE_WH_KERNELS_FOR_BLOCK_SIZE(qual, scalar_t, NB, TDOF) \
    DECLARE_WH_KERNEL(qual, scalar_t, NB, TDOF, 256)                \
    DECLARE_WH_KERNEL(qual, scalar_t, NB, TDOF, 512)                \
    DECLARE_WH_KERNEL(qual, scalar_t, NB, TDOF, 1024)

#define DECLARE_WH_KERNELS_FOR_TDOF(qual, scalar_t, NB)      \
    DECLARE_WH_KERNELS_FOR_BLOCK_SIZE(qual, scalar_t, NB, 1) \
    DECLARE_WH_KERNELS_FOR_BLOCK_SIZE(qual, scalar_t, NB, 2) \
    DECLARE_WH_KERNELS_FOR_BLOCK_SIZE(qual, scalar_t, NB, 3) \
    DECLARE_WH_KERNELS_FOR_BLOCK_SIZE(qual, scalar_t, NB, 4)

#define DECLARE_WH_KERNELS_FOR_NB(qual, scalar_t)  \
    DECLARE_WH_KERNELS_FOR_TDOF(qual, scalar_t, 1) \
    DECLARE_WH_KERNELS_FOR_TDOF(qual, scalar_t, 2) \
    DECLARE_WH_KERNELS_FOR_TDOF(qual, scalar_t, 3) \
    DECLARE_WH_KERNELS_FOR_TDOF(qual, scalar_t, 4) \
    DECLARE_WH_KERNELS_FOR_TDOF(qual, scalar_t, 5) \
    DECLARE_WH_KERNELS_FOR_TDOF(qual, scalar_t, 6) \
    DECLARE_WH_KERNELS_FOR_TDOF(qual, scalar_t, 7) \
    DECLARE_WH_KERNELS_FOR_TDOF(qual, scalar_t, 8)

namespace cuddh::details
{
    template <typename scalar_t>
    struct WHKernelDispatcher
    {
        int n_basis, tdof, block_size;
        WHKernelDispatcher(int n_basis, int tdof, int block_size) : n_basis(n_basis), tdof(tdof), block_size(block_size)
        {}

        template <int NB, int TDOF, int BLOCK_SIZE>
        static void dispatch_kernel(const EnsembleSpace &efem, int g_ndof, int n_lambda,
                                    const LambdaDOFData<scalar_t> *B, const DDStiffnessMatrix<scalar_t> &S,
                                    const scalar_t *punity, const DDWaveHoltz<scalar_t> &W, int wh_iterations,
                                    const double *fem_in, double *fem_out, const scalar_t *lambda_in,
                                    scalar_t *lambda_out, scalar_t *d_work);

        template <int NB, int TDOF, typename... Args>
        void dispatch_blocksize(Args &&...args) const
        {
            switch (block_size)
            {
                case 256:
                    dispatch_kernel<NB, TDOF, 256>(std::forward<Args>(args)...);
                    break;
                case 512:
                    dispatch_kernel<NB, TDOF, 512>(std::forward<Args>(args)...);
                    break;
                case 1024:
                    dispatch_kernel<NB, TDOF, 1024>(std::forward<Args>(args)...);
                    break;
                default:
                    cuddh_verify(false, printf("DDH error: block_size (=%d) not supported.\n", block_size));
            }
        }

        template <int NB, typename... Args>
        void dispatch_tdof(Args &&...args) const
        {
            switch (tdof)
            {
                case 1:
                    dispatch_blocksize<NB, 1>(std::forward<Args>(args)...);
                    break;
                case 2:
                    dispatch_blocksize<NB, 2>(std::forward<Args>(args)...);
                    break;
                case 3:
                    dispatch_blocksize<NB, 3>(std::forward<Args>(args)...);
                    break;
                case 4:
                    dispatch_blocksize<NB, 4>(std::forward<Args>(args)...);
                    break;
                default:
                    cuddh_verify(false, printf("DDH error: only tdof (=%d) <= 4\n", tdof));
            }
        }

        template <typename... Args>
        void invoke(Args &&...args) const
        {
            switch (n_basis)
            {
                case 2:
                    dispatch_tdof<2>(std::forward<Args>(args)...);
                    break;
                case 3:
                    dispatch_tdof<3>(std::forward<Args>(args)...);
                    break;
                case 4:
                    dispatch_tdof<4>(std::forward<Args>(args)...);
                    break;
                case 5:
                    dispatch_tdof<5>(std::forward<Args>(args)...);
                    break;
                case 6:
                    dispatch_tdof<6>(std::forward<Args>(args)...);
                    break;
                case 7:
                    dispatch_tdof<7>(std::forward<Args>(args)...);
                    break;
                case 8:
                    dispatch_tdof<8>(std::forward<Args>(args)...);
                    break;
                default:
                    cuddh_verify(false, printf("DDH error: only n_basis (=%d) <= 8 supported\n", n_basis));
            }
        }
    };

    DECLARE_WH_KERNELS_FOR_NB(extern, float)
    DECLARE_WH_KERNELS_FOR_NB(extern, double)
} // namespace cuddh::details