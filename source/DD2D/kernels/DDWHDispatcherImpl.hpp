#pragma once

#include "DDWHDispatcher.hpp"
#include "DDWHKernel.hpp"

namespace cuddh::details
{
    template <typename scalar_t>
    template <int NB, int TDOF, int BLOCK_SIZE>
    void WHKernelDispatcher<scalar_t>::dispatch_kernel(const EnsembleSpace &efem, int g_ndof, int n_lambda,
                                                       const LambdaDOFData<scalar_t> *B,
                                                       const DDStiffnessMatrix<scalar_t> &S, const scalar_t *punity,
                                                       const DDWaveHoltz<scalar_t> &W, int wh_iterations,
                                                       const double *fem_in, double *fem_out, const scalar_t *lambda_in,
                                                       scalar_t *lambda_out, scalar_t *d_work)
    {
        constexpr int NEL = BLOCK_SIZE / (NB * NB);

        if (fem_out)
            dla::zeros(2 * g_ndof, fem_out);
        if (lambda_out)
            dla::zeros(2 * n_lambda, lambda_out);

        auto data = DDHWHKernelData<scalar_t, NB, NEL, TDOF>::make(n_lambda, g_ndof, efem, B, punity, S, W,
                                                                   wh_iterations, d_work);
        const int n_domains = efem.size();
        dim3 bs(NB * NB, NEL);
        ddh_wh_action_kernel<scalar_t, NB, NEL, TDOF><<<n_domains, bs>>>(data, fem_in, fem_out, lambda_in, lambda_out);
        CUDDH_CHECK_KERNEL();
    }
} // namespace cuddh::details