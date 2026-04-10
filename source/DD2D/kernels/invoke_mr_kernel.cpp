#include "source/DD2D/kernels/DDMRDispatcher.hpp"
#include "source/DD2D/kernels/DDMRKernel.hpp"

using namespace cuddh;

namespace cuddh::details
{
    template <typename scalar_t>
    void invoke_mr_kernel(int n_basis, int tdof, int block_size, const EnsembleSpace &efem, int g_ndof, int n_lambda,
                          const LambdaDOFData<scalar_t> *B, const DDStiffnessMatrix<scalar_t> &S,
                          const scalar_t *punity, MatrixWrapper<const scalar_t> scaled_mass,
                          MatrixWrapper<const scalar_t> scaled_face_mass, scalar_t omega, const double *fem_in,
                          double *fem_out, const scalar_t *lambda_in, scalar_t *lambda_out, scalar_t *d_work)
    {
        MRKernelDispatcher<scalar_t>(n_basis, tdof, block_size)
            .invoke(efem, g_ndof, n_lambda, B, S, punity, scaled_mass, scaled_face_mass, omega, fem_in, fem_out,
                    lambda_in, lambda_out, d_work);
    }

    template void invoke_mr_kernel<float>(int, int, int, const EnsembleSpace &, int, int, const LambdaDOFData<float> *,
                                          const DDStiffnessMatrix<float> &, const float *, MatrixWrapper<const float>,
                                          MatrixWrapper<const float>, float, const double *, double *, const float *,
                                          float *, float *);

    template void invoke_mr_kernel<double>(int, int, int, const EnsembleSpace &, int, int,
                                           const LambdaDOFData<double> *, const DDStiffnessMatrix<double> &,
                                           const double *, MatrixWrapper<const double>, MatrixWrapper<const double>,
                                           double, const double *, double *, const double *, double *, double *);
} // namespace cuddh::details
