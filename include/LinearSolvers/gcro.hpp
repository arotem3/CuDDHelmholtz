#pragma once
#include "LinearSolvers/KrylovHelpers.hpp"
#include "LinearSolvers/SolverBase.hpp"

namespace cuddh
{
    struct GCROParams
    {
        int maxit = 100;    // maximum number of iterations
        double rtol = 1e-3; // relative tolerance for an acceptable solution. GCRO stops when |A*x-b|/|b| < tol.
        double atol = 0.0;  // absolute tolerance for an acceptable solution. GCRO stops when |A*x-b| < atol.
        SolverVerbosity verbose = SolverVerbosity::Silent; // 0: silent, 1: progress bar, 2: one line per iteration
    };

    /**
     * @brief Flexible GCRO-DR solver.
     *
     * @tparam real_t
     */
    template <typename real_t>
    class GCRO
    {
    public:
        GCRO(int _n, Operator<double> &_A, Operator<double> *_M = nullptr, int _kdim = 40, int _edim = 20)
            : n{_n},
              kdim{_kdim},
              edim{_edim},
              active_edim{0},
              A{&_A},
              Prec{_M},
              update_deflation{true},
              _W(n * (kdim + 1)),
              _Z(n * kdim),
              _r(n),
              H(kdim + 1, kdim),
              y(kdim + 1),
              cs(kdim),
              sn(kdim)
        {
            cuddh_verify(n >= 0, printf("GCRO: n must be non-negative"));
            cuddh_verify(edim >= 0, printf("GCRO: edim must be non-negative"));
            cuddh_verify(kdim > edim + 1, printf("GCRO: kdim must be greater than edim+1"));
            cublasCreate(&cublas_handle);
        }

        ~GCRO() { cublasDestroy(cublas_handle); }

        constexpr void toggle_deflation_update(bool update) { update_deflation = update; }

        constexpr void reset_deflation() { active_edim = 0; }

        SolverResults solve(real_t *x, const real_t *b, GCROParams opts = {}) const
        {
            cuddh_verify(opts.maxit >= 0, printf("GCRO: maxit must be non-negative"));
            cuddh_verify(opts.rtol >= 0, printf("GCRO: rtol must be non-negative"));
            cuddh_verify(opts.atol >= 0, printf("GCRO: atol must be non-negative"));

            SolverLogger logger(opts.verbose, opts.maxit);

            const real_t bnrm = dla::norm(n, b);
            const real_t tol = std::max(opts.rtol * bnrm, opts.atol);

            auto r = reshape(_r, n);
            A->action(x, r); // r <- A * x
            logger.log_matvec();
            dla::axpby(n, 1.0, b, -1.0, r); // r <- b - r = b - A * x

            // apply deflation correction
            for (int i = 0; i < active_edim; ++i)
            {
                auto w = thrust::raw_pointer_cast(_W.data()) + n * i;
                auto z = thrust::raw_pointer_cast(_Z.data()) + n * i;
                real_t alpha = dla::dot(n, w, r);
                dla::axpby(n, alpha, z, real_t(1.0), x);  // x += (w, r) * z
                dla::axpby(n, -alpha, w, real_t(1.0), r); // r -= (w, r) * w
            }

            real_t rnrm = dla::norm(n, r);

            logger.log_iteration(rnrm / bnrm);

            while (logger.num_iterations() < opts.maxit && rnrm > tol)
            {
                int m = std::min(kdim, opts.maxit - logger.num_iterations());
                m = arnoldi(logger, x, r, rnrm, bnrm, m, tol);

                if (update_deflation)
                    compute_deflation_ritz_vecs(m);
            }

            return logger.log_summary(rnrm / bnrm, rnrm <= tol);
        }

    private:
        int arnoldi(SolverLogger &logger, real_t *x, real_t *r, real_t &rnrm, real_t bnrm, int m, real_t tol) const
        {
            constexpr real_t zero(0.0), one(1.0), eps{std::is_same_v<real_t, float> ? 1e-7 : 1e-14};
            const int k = active_edim;
            m = std::min(m, kdim - k);

            cuddh_verify(m + k <= kdim, printf("GCRO: Arnoldi m+k exceeds kdim"));

            auto W = thrust::raw_pointer_cast(_W.data());
            auto Z = thrust::raw_pointer_cast(_Z.data());

            std::fill(H.begin(), H.end(), zero);
            std::fill(y.begin(), y.end(), zero);

            for (int i = 0; i < k; ++i)
                H(i, i) = one;

            dla::axpby(n, one / rnrm, r, zero, W + n * k); // w[k] <- r / ||r||
            y[k] = rnrm;

            int actual_kdim = 0;
            for (int j = 0; j < m; ++j)
            {
                const int col = k + j;

                real_t *w = W + n * col;
                real_t *z = Z + n * col;

                if (Prec)
                    Prec->action(w, z);
                else
                    dla::copy(n, w, z);

                w = W + n * (col + 1);
                A->action(z, w);
                logger.log_matvec();

                // apply MGS to deflation and krylov vectors (both in W)
                for (int i = 0; i <= col; ++i)
                {
                    const real_t *v = W + n * i;
                    real_t hij = dla::dot(n, v, w);
                    H(i, col) = hij;
                    dla::axpby(n, -hij, v, one, w); // w -= (v, w) * v
                }

                real_t wnrm = dla::norm(n, w);
                H(col + 1, col) = wnrm;

                bool breakdown = wnrm < eps;
                if (breakdown)
                    dla::zeros(n, w);
                else
                    dla::scal(n, one / wnrm, w); // w /= ||w||

                // Update QR of H and RHS via Givens rotations
                for (int i = 0; i < j; ++i)
                    apply_givens(H(k + i, col), H(k + i + 1, col), cs[i], sn[i]);
                std::tie(cs[j], sn[j]) = compute_givens(H(col, col), H(col + 1, col));
                apply_givens(H(col, col), H(col + 1, col), cs[j], sn[j]);
                apply_givens(y[col], y[col + 1], cs[j], sn[j]);

                rnrm = std::abs(y[col + 1]);
                logger.log_iteration(rnrm / bnrm);

                actual_kdim = j + 1;
                if (rnrm <= tol || breakdown)
                    break;
            }

            // Apply Q (QR of H) to W
            for (int j = 0; j < actual_kdim; ++j)
            {
                real_t *w1 = W + n * (k + j);
                real_t *w2 = w1 + n;
                rotate_vecs(n, w1, w2, cs[j], sn[j]);
            }

            int ncol = k + actual_kdim;

            solve_triu(ncol, H.data(), H.shape(0), y.data()); // solve R * y = Q^T * f

            for (int i = 0; i < ncol; ++i)
                dla::axpby(n, y[i], Z + n * i, one, x);    // x += y[i] * Z[:, i]
            dla::axpby(n, y[ncol], W + n * ncol, zero, r); // r = y[ncol] * W[:, ncol]

            return ncol;
        }

        // Update W and Z with new deflation vectors. m is the Krylov dimension of the last cycle
        void compute_deflation_ritz_vecs(int m) const
        {
            constexpr real_t zero(0.0), one(1.0);
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

    private:
        const int n;    // dimension of x
        const int kdim; // total Krylov dimension
        const int edim; // desired deflation dimension

        mutable int active_edim; // current deflation dimension

        const Operator<double> *A;
        const Operator<double> *Prec;

        bool update_deflation;

        mutable thrust::device_vector<real_t> _W; // Krylov basis vectors, (n, kdim+1)
        mutable thrust::device_vector<real_t> _Z; // Preconditioned Krylov basis vectors, (n, kdim)
        mutable thrust::device_vector<real_t> _r; // residual vector, (n)

        mutable Matrix<real_t> H; // Upper Hessenberg matrix, (kdim+1, kdim)
        mutable Vec<real_t> y;    // RHS/Solution of the least squares problem, (kdim+1)

        mutable Vec<real_t> cs; // cosines for Givens rotations, (kdim)
        mutable Vec<real_t> sn; // sines for Givens rotations, (kdim)

        cublasHandle_t cublas_handle; // Persistent cuBLAS handle
    };
} // namespace cuddh
