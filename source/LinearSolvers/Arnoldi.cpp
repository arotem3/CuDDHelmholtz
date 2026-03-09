#include "LinearSolvers/Arnoldi.hpp"

using namespace cuddh;

template <typename real_t>
int BaseArnoldiSolver<real_t>::arnoldi_cycle(SolverLogger &logger, int m, int k, real_t *x, real_t *r, real_t &rnrm,
                                             real_t bnrm, real_t tol) const
{
    constexpr real_t zero(0.0), one(1.0), eps{std::is_same_v<real_t, float> ? 1e-7 : 1e-14};

    cuddh_verify(0 <= k && k <= kdim,
                 printf("The augmented dimension k = %d must be in the range [0, kdim = %d]\n", k, kdim));
    cuddh_verify(0 <= m && m <= kdim, printf("Krylov dimension m = %d must be in the range [0, kdim = %d]\n", m, kdim));

    m = std::min(m, kdim - k);

    real_t *W = thrust::raw_pointer_cast(_W.data());
    real_t *Z = thrust::raw_pointer_cast(_Z.data());

    std::fill(H.begin(), H.end(), zero);
    std::fill(eta.begin(), eta.end(), zero);

    for (int i = 0; i < k; ++i)
        H(i, i) = one;

    dla::axpby(n, one / rnrm, r, zero, W + n * k);
    eta[k] = rnrm;

    int m1 = 0;
    for (int j = 0; j < m; ++j)
    {
        m1 = j + 1;
        const int col = k + j;

        real_t *w = W + n * col;

        if (flexible)
        {
            real_t *z = Z + n * col;

            if (M)
                M->action(w, z);
            else
                dla::copy(n, w, z);

            w = w + n;
            A->action(z, w);
        }
        else
        {
            real_t *w1 = w + n;

            if (M)
            {
                A->action(w, Z);
                M->action(Z, w1);
            }
            else
                A->action(w, w1);

            w = w1;
        }

        logger.log_matvec();

        // MGS
        for (int i = 0; i <= col; ++i)
        {
            const real_t *v = W + n * i;
            real_t hij = dla::dot(n, v, w);
            H(i, col) = hij;
            dla::axpby(n, -hij, v, one, w); // w -= (v, w) * v
        }

        real_t wnrm = dla::norm(n, w);
        H(col + 1, col) = wnrm;

        if (wnrm < eps)
        {
            m1--; // discard iteration
            break;
        }

        dla::scal(n, one / wnrm, w); // w /= ||w||

        // update QR
        for (int i = 0; i < j; ++i)
            apply_givens(H(k + i, col), H(k + i + 1, col), cs(i), sn(i));
        std::tie(cs(j), sn(j)) = compute_givens(H(col, col), H(col + 1, col));
        apply_givens(H(col, col), H(col + 1, col), cs(j), sn(j));
        apply_givens(eta[col], eta[col + 1], cs(j), sn(j));

        rnrm = std::abs(eta[col + 1]);
        logger.log_iteration(rnrm / bnrm);

        if (rnrm <= tol)
            break;
    }

    // Apply givens to W
    if (flexible)
        for (int j = 0; j < m1; ++j)
        {
            real_t *w1 = W + n * (k + j);
            real_t *w2 = w1 + n;
            rotate_vecs(n, w1, w2, cs(j), sn(j));
        }

    // solve LS
    int ncol = k + m1;
    solve_triu(ncol, H.data(), H.shape(0), eta.data());

    // update solution and residual
    real_t *V = (flexible) ? Z : W;
    for (int i = 0; i < ncol; ++i)
        dla::axpby(n, eta[i], V + n * i, one, x); // x += y[i] * Z[:, i]

    if (flexible)
        dla::axpby(n, eta[ncol], W + n * ncol, zero, r); // r = y[ncol] * W[:, ncol]

    return ncol;
}

template <typename real_t>
BaseArnoldiSolver<real_t>::BaseArnoldiSolver(int n, const Operator<real_t> &A, const Operator<real_t> *M, int kdim,
                                             bool flexible)
    : n{n},
      kdim{kdim <= n ? kdim : n},
      flexible{flexible},
      A{&A},
      M{M},
      _W(n * (kdim + 1)),
      H(kdim + 1, kdim),
      eta(kdim + 1),
      cs(kdim),
      sn(kdim)
{
    cuddh_verify(n >= 0, printf("solver error: n = %d must be non-negative.\n", n));
    cuddh_verify(kdim >= 0, printf("solver error: kdim = %d must be non-negative.\n", kdim));

    if (flexible)
        _Z.resize(n * kdim);
}

template class BaseArnoldiSolver<float>;
template class BaseArnoldiSolver<double>;
