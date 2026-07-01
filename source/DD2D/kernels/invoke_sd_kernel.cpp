#ifdef CUDDH_HAS_CUDSS

#include "LambdaDOFData.hpp"
#include "SmallMatrix.hpp"
#include "SparseMatrix.hpp"

using namespace cuddh;

namespace
{
    // Gather: reads global FEM + lambda DOFs into a contiguous blocked-complex RHS buffer.
    // Writes conj(F) = (F.x, -F.y) so that SparseBlockLU solves A·y = conj(F),
    // matching the convention of the MINRES subdomain solver (which solves L·u = (F.x, -F.y)
    // via the real symmetric block form of the Helmholtz system).
    template <typename scalar_t>
    __global__ void ddh_sd_gather_kernel(int mx_ndof, int mx_fdof, int g_ndof, int n_lambda,
                                         const int *__restrict__ s_ndof, const int *__restrict__ s_fdof,
                                         const int *__restrict__ gI, const scalar_t *__restrict__ punity,
                                         const LambdaDOFData<scalar_t> *__restrict__ B,
                                         const double *__restrict__ fem_in, const scalar_t *__restrict__ lambda_in,
                                         scalar_t *__restrict__ d_rhs)
    {
        using vec_t = scalar2<scalar_t>;
        const int subsp = blockIdx.x;
        const int ndof = s_ndof[subsp];
        const int fdof = s_fdof[subsp];

        for (int i = static_cast<int>(threadIdx.x); i < ndof; i += static_cast<int>(blockDim.x))
        {
            vec_t F{};

            if (fem_in)
            {
                const int g_idx = gI[i + mx_ndof * subsp];
                const scalar_t w = punity[i + mx_ndof * subsp];
                F.x = w * static_cast<scalar_t>(fem_in[g_idx]);
                F.y = w * static_cast<scalar_t>(fem_in[g_ndof + g_idx]);
            }

            if (lambda_in && i < fdof)
            {
                for (int o = 0; o < 2; ++o)
                {
                    const auto [li, lj, T] = B[o + 2 * i + 2 * mx_fdof * subsp];
                    if (li < 0)
                        break;

                    scalar_t lam = lambda_in[li];
                    scalar_t re = lam, im = lam;

                    lam = lambda_in[lj];
                    re += lam;
                    im -= lam;

                    lam = lambda_in[n_lambda + li];
                    re -= lam;
                    im += lam;

                    lam = lambda_in[n_lambda + lj];
                    re += lam;
                    im += lam;

                    re *= scalar_t(0.5);
                    im *= scalar_t(0.5);

                    F.x += T * re;
                    F.y += T * im;
                }
            }

            d_rhs[subsp * 2 * mx_ndof + i] = F.x;
            d_rhs[subsp * 2 * mx_ndof + mx_ndof + i] = -F.y;
        }
    }

    // Scatter: maps per-subdomain CuDSS solutions back to global FEM output and lambda update.
    template <typename scalar_t>
    __global__ void ddh_sd_scatter_kernel(int mx_ndof, int mx_fdof, int g_ndof, int n_lambda,
                                          const int *__restrict__ s_ndof, const int *__restrict__ s_fdof,
                                          const int *__restrict__ gI, const scalar_t *__restrict__ punity,
                                          const LambdaDOFData<scalar_t> *__restrict__ B,
                                          const scalar_t *__restrict__ d_sol, double *__restrict__ fem_out,
                                          const scalar_t *__restrict__ lambda_in, scalar_t *__restrict__ lambda_out)
    {
        const int subsp = blockIdx.x;
        const int ndof = s_ndof[subsp];
        const int fdof = s_fdof[subsp];

        for (int i = static_cast<int>(threadIdx.x); i < ndof; i += static_cast<int>(blockDim.x))
        {
            const scalar_t sol_re = d_sol[subsp * 2 * mx_ndof + i];
            const scalar_t sol_im = d_sol[subsp * 2 * mx_ndof + mx_ndof + i];

            if (fem_out)
            {
                const int g_idx = gI[i + mx_ndof * subsp];
                const scalar_t w = punity[i + mx_ndof * subsp];
                atomicAdd(fem_out + g_idx, double(w * sol_re));
                atomicAdd(fem_out + g_ndof + g_idx, double(w * sol_im));
            }

            if (lambda_out && i < fdof)
            {
                for (int o = 0; o < 2; ++o)
                {
                    const auto [li, lj, T] = B[o + 2 * i + 2 * mx_fdof * subsp];
                    if (li < 0)
                        break;

                    scalar_t lam = 0, mu = 0;
                    if (lambda_in)
                    {
                        lam = lambda_in[li];
                        mu = lambda_in[n_lambda + li];
                    }

                    lambda_out[lj] = -lam + T * sol_im;
                    lambda_out[n_lambda + lj] = -mu - T * sol_re;
                }
            }
        }
    }
} // anonymous namespace

namespace cuddh::details
{
    template <typename scalar_t>
    void invoke_sd_kernel(int n_domains, int mx_ndof, int mx_fdof, int g_ndof, int n_lambda, const int *s_ndof,
                          const int *s_fdof, const int *gI, const scalar_t *punity, const LambdaDOFData<scalar_t> *B,
                          const double *fem_in, double *fem_out, const scalar_t *lambda_in, scalar_t *lambda_out,
                          SparseBlockLU<scalar_t, true> &lu, scalar_t *d_rhs, scalar_t *d_sol)
    {
        const int block_size = 256;

        ddh_sd_gather_kernel<<<n_domains, block_size>>>(mx_ndof, mx_fdof, g_ndof, n_lambda, s_ndof, s_fdof, gI, punity,
                                                        B, fem_in, lambda_in, d_rhs);

        lu.solve(d_rhs, d_sol);

        ddh_sd_scatter_kernel<<<n_domains, block_size>>>(mx_ndof, mx_fdof, g_ndof, n_lambda, s_ndof, s_fdof, gI, punity,
                                                         B, d_sol, fem_out, lambda_in, lambda_out);
    }

    template void invoke_sd_kernel<float>(int, int, int, int, int, const int *, const int *, const int *, const float *,
                                          const LambdaDOFData<float> *, const double *, double *, const float *,
                                          float *, SparseBlockLU<float, true> &, float *, float *);

    template void invoke_sd_kernel<double>(int, int, int, int, int, const int *, const int *, const int *,
                                           const double *, const LambdaDOFData<double> *, const double *, double *,
                                           const double *, double *, SparseBlockLU<double, true> &, double *, double *);
} // namespace cuddh::details

#endif // CUDDH_HAS_CUDSS
