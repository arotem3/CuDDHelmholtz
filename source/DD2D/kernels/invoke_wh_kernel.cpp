#include "source/DD2D/kernels/DDWHDispatcher.hpp"
#include "source/DD2D/kernels/DDWHKernel.hpp"

using namespace cuddh;

namespace cuddh::details
{
    template <typename scalar_t>
    void invoke_wh_kernel(int n_basis, int tdof, int block_size, const EnsembleSpace &efem, int g_ndof, int n_lambda,
                          const LambdaDOFData<scalar_t> *B, const DDStiffnessMatrix<scalar_t> &S,
                          const scalar_t *punity, const DDWaveHoltz<scalar_t> &W, int waveholtz_iterations,
                          const double *fem_in, double *fem_out, const scalar_t *lambda_in, scalar_t *lambda_out,
                          scalar_t *d_work)
    {
        WHKernelDispatcher<scalar_t>(n_basis, tdof, block_size)
            .invoke(efem, g_ndof, n_lambda, B, S, punity, W, waveholtz_iterations, fem_in, fem_out, lambda_in,
                    lambda_out, d_work);
    }

    template void invoke_wh_kernel<float>(int, int, int, const EnsembleSpace &, int, int, const LambdaDOFData<float> *,
                                          const DDStiffnessMatrix<float> &, const float *, const DDWaveHoltz<float> &,
                                          int, const double *, double *, const float *, float *, float *);

    template void invoke_wh_kernel<double>(int, int, int, const EnsembleSpace &, int, int,
                                           const LambdaDOFData<double> *, const DDStiffnessMatrix<double> &,
                                           const double *, const DDWaveHoltz<double> &, int, const double *, double *,
                                           const double *, double *, double *);
} // namespace cuddh::details
