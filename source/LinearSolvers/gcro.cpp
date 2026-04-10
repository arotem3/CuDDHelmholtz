#include "LinearSolvers/gcro.hpp"

using namespace cuddh;

template <typename real_t>
GCRO<real_t>::GCRO(Operator<real_t> &A, Operator<real_t> *M, int kdim, int edim)
    : Solver<real_t>(A.ndof()),
      BaseArnoldiSolver<real_t>(A, M, kdim, true),
      edim{edim},
      active_edim{0},
      update_deflation{true},
      _r(A.ndof())
{
    cuddh_verify(!M || M->ndof() == A.ndof(),
                 printf("GCRO: preconditioner M.ndof(=%d) must equal operator A.ndof(=%d)\n", M->ndof(), A.ndof()));
    cuddh_verify(edim >= 0, printf("GCRO: edim must be non-negative\n"));
    cuddh_verify(kdim > edim + 1, printf("GCRO: kdim must be greater than edim+1\n"));
    cublasCreate(&cublas_handle);
}

template <typename real_t>
SolverResults GCRO<real_t>::solve(real_t *x, const real_t *b, SolverParams opts) const
{
    validate_params(opts);

    SolverLogger logger(opts.verbose, opts.maxit);

    const int n = this->ndof();

    const real_t bnrm = dla::norm(n, b);
    const real_t tol = std::max(opts.rtol * bnrm, opts.atol);

    auto r = thrust::raw_pointer_cast(_r.data());

    this->evaluate_residual(r, x, b);
    logger.log_matvec();

    deflate(x, r);

    real_t rnrm = dla::norm(n, r);
    logger.log_iteration(rnrm / bnrm);

    while (logger.num_iterations() < opts.maxit && rnrm > tol)
    {
        int m = std::min(kdim, opts.maxit - logger.num_iterations());
        m = this->arnoldi_cycle(logger, m, active_edim, x, r, rnrm, bnrm, tol);

        this->evaluate_residual(r, x, b);
        logger.log_matvec();
        rnrm = dla::norm(n, r);

        if (update_deflation)
            compute_deflation_ritz_vecs(m);
    }

    return logger.log_summary(rnrm / bnrm, rnrm <= tol);
}

template <typename real_t>
void GCRO<real_t>::deflate(real_t *x, real_t *r) const
{
    const int n = this->ndof();
    auto W = thrust::raw_pointer_cast(_W.data());
    auto Z = thrust::raw_pointer_cast(_Z.data());

    for (int i = 0; i < active_edim; ++i)
    {
        auto w = W + n * i;
        auto z = Z + n * i;
        real_t alpha = dla::dot(n, w, r);
        dla::axpby(n, alpha, z, real_t(1.0), x);  // x += (w, r) * z
        dla::axpby(n, -alpha, w, real_t(1.0), r); // r -= (w, r) * w
    }
}

template <typename real_t>
void GCRO<real_t>::compute_deflation_ritz_vecs(int m) const
{
    constexpr real_t zero(0.0), one(1.0);
    const int n = this->ndof();
    const int k = std::min(m, edim);

    // X = H[:m+1, :m]' * H[:m+1, :m]
    Matrix<real_t> X(m, m);
    int ldh = H.shape(0); // leading dimension of H (kdim+1)
    hla::gemm(m, m, m + 1, one, H.data(), ldh, H.data(), ldh, zero, X.data(), m, true, false);

    // S = W[:, :m+1]' * Z[:, :m]
    // Use cuBLAS for device matrix multiplication
    auto WZ = [&]() -> thrust::host_vector<real_t> {
        thrust::device_vector<real_t> S_dev((m + 1) * m);
        const real_t *d_W = thrust::raw_pointer_cast(_W.data());
        const real_t *d_Z = thrust::raw_pointer_cast(_Z.data());
        real_t *d_S = thrust::raw_pointer_cast(S_dev.data());
        dla::gemm(cublas_handle, m + 1, m, n, one, d_W, n, d_Z, n, zero, d_S, m + 1, true, false);
        thrust::host_vector<real_t> S_host = S_dev;
        return S_host;
    }();

    // Y = H[:m+1, :m]' * WZ
    Matrix<real_t> Y(m, m);
    hla::gemm(m, m, m + 1, one, H.data(), ldh, WZ.data(), m + 1, zero, Y.data(), m, true, false);

    // Real QZ + invariant subspace extraction.
    // X and Y are overwritten in-place with the Schur form
    Matrix<real_t> P = qz_invariant_space(m, k, X.data(), m, Y.data(), m);
    active_edim = P.shape(1);

    // HP = H[:m+1, :m] * P[:, :active_edim]
    Matrix<real_t> HP(m + 1, active_edim);
    hla::gemm(m + 1, active_edim, m, one, H.data(), ldh, P.data(), m, zero, HP.data(), m + 1);

    // Compute QR decomposition of HP in place: HP = Q * R
    std::vector<real_t> tau_W;
    qr_decomp(m + 1, active_edim, HP.data(), m + 1, tau_W);

    // W[:, :m+1] = W[:, :m+1] * Q using Householder reflectors on device
    {
        real_t *d_W = thrust::raw_pointer_cast(_W.data());

        apply_q_right_device(cublas_handle, n, m + 1, active_edim, HP.data(), m + 1, tau_W.data(), d_W, n);
    }

    // Z[:, :active_edim] = Z[:, :m] * P * R^{-1}
    // Compute P * R^{-1} in place, then QR, then apply to Z
    {
        // P = P * R^{-1}, using HP as R (upper triangular)
        trsm_upper_right(m, active_edim, one, HP.data(), m + 1, P.data(), m);

        // Compute QR of P[:m, :active_edim] in place
        std::vector<real_t> tau_Z;
        qr_decomp(m, active_edim, P.data(), m, tau_Z);

        real_t *d_Z = thrust::raw_pointer_cast(_Z.data());

        // Z[:, :m] = Z[:, :m] * Q, where Q is stored in P with tau_Z
        apply_q_right_device(cublas_handle, n, m, active_edim, P.data(), m, tau_Z.data(), d_Z, n);

        // Z[:, :active_edim] = Z[:, :active_edim] * R, where R is upper triangular part of P
        trmm_right_device(cublas_handle, n, active_edim, one, P.data(), m, d_Z, n);
    }
}

template class GCRO<float>;
template class GCRO<double>;
