#include "LinearSolvers/KrylovHelpers.hpp"

using namespace cuddh;

template <typename real_t>
static inline void _rotate_vecs(int n, real_t *d_X, real_t *d_Y, real_t cs, real_t sn)
{
    forall(n, [=] __device__(int i) {
        real_t x = d_X[i], y = d_Y[i];

        d_X[i] = cs * x + sn * y;
        d_Y[i] = -sn * x + cs * y;
    });
}

void cuddh::rotate_vecs(int n, double *d_X, double *d_Y, double cs, double sn)
{
    _rotate_vecs<double>(n, d_X, d_Y, cs, sn);
}

void cuddh::rotate_vecs(int n, float *d_X, float *d_Y, float cs, float sn)
{
    _rotate_vecs<float>(n, d_X, d_Y, cs, sn);
}

template <typename scalar_t, typename... Args>
static void gges(Args &&...args)
{
    static_assert(std::is_same_v<scalar_t, double> || std::is_same_v<scalar_t, float>,
                  "gges implemented only for float and double.");

    lapack_int info;
    if constexpr (std::is_same_v<scalar_t, double>)
        info = LAPACKE_dgges(std::forward<Args>(args)...);
    else if constexpr (std::is_same_v<scalar_t, float>)
        info = LAPACKE_sgges(std::forward<Args>(args)...);

    cuddh_verify(info == 0, {
        if (info < 0)
            printf("LAPACKE_dgges: invalid argument index %d\n", -info);
        else
            printf("LAPACKE_dgges: QZ algorithm failed to converge (info=%d)\n", info);
    });
}

template <typename scalar_t>
static Matrix<scalar_t> _qz_invariant_space(int n, int nkeep, scalar_t *A, int lda, scalar_t *B, int ldb)
{
    cuddh_verify(0 <= nkeep && nkeep <= n, printf("qz_invariant_space: nkeep must be in [0,n]"));
    cuddh_verify(lda >= n && ldb >= n, printf("qz_invariant_space: leading dimensions must be at least n"));

    Matrix<scalar_t> Q(n, n);
    int ldq = Q.shape(0);

    std::vector<scalar_t> alphar(n), alphai(n), beta(n);
    lapack_int sdim = 0;
    gges<scalar_t>(LAPACK_COL_MAJOR, 'V', 'N', 'N', nullptr, n, A, lda, B, ldb, &sdim, alphar.data(), alphai.data(),
                   beta.data(), Q.data(), ldq, nullptr, 1);

    if (nkeep == 0)
        return Matrix<scalar_t>(0, 0);

    struct Block
    {
        int first;
        int size;
        scalar_t key;
    };

    auto eig_mag = [&](int i) {
        const scalar_t b = std::abs(beta[i]);
        if (b <= std::numeric_limits<scalar_t>::epsilon())
            return std::numeric_limits<scalar_t>::infinity();
        return std::hypot(alphar[i], alphai[i]) / b;
    };

    const double eps = 64.0 * std::numeric_limits<scalar_t>::epsilon();
    std::vector<Block> blocks;
    for (int i = 0; i < n;)
    {
        const bool is_pair = (i + 1 < n) && (std::abs(alphai[i]) > eps) && (std::abs(alphai[i] + alphai[i + 1]) <= eps);
        if (is_pair)
        {
            blocks.push_back({i, 2, std::min(eig_mag(i), eig_mag(i + 1))});
            i += 2;
        }
        else
        {
            blocks.push_back({i, 1, eig_mag(i)});
            i += 1;
        }
    }

    std::sort(blocks.begin(), blocks.end(), [](const Block &a, const Block &b) { return a.key < b.key; });

    std::vector<scalar_t> cols;
    cols.reserve(nkeep + 1);
    for (const auto &blk : blocks)
    {
        if ((int)cols.size() >= nkeep)
            break;
        cols.push_back(blk.first);
        if (blk.size == 2)
            cols.push_back(blk.first + 1);
    }

    std::sort(cols.begin(), cols.end());

    int nsel = (int)cols.size();
    Matrix<scalar_t> Qsel(n, nsel);

    for (int j = 0; j < nsel; ++j)
    {
        const int src_col = cols[j];
        for (int i = 0; i < n; ++i)
            Qsel(i, j) = Q(i, src_col);
    }

    return Qsel;
}

Matrix<float> cuddh::qz_invariant_space(int n, int nkeep, float *A, int lda, float *B, int ldb)
{
    return _qz_invariant_space<float>(n, nkeep, A, lda, B, ldb);
}

Matrix<double> cuddh::qz_invariant_space(int n, int nkeep, double *A, int lda, double *B, int ldb)
{
    return _qz_invariant_space<double>(n, nkeep, A, lda, B, ldb);
}

template <typename scalar_t, typename... Args>
static void cu_gemv(Args &&...args)
{
    static_assert(std::is_same_v<scalar_t, double> || std::is_same_v<scalar_t, float>,
                  "cu_gemv implemented only for float and double.");

    if constexpr (std::is_same_v<scalar_t, double>)
        cublasDgemv(std::forward<Args>(args)...);
    else if constexpr (std::is_same_v<scalar_t, float>)
        cublasSgemv(std::forward<Args>(args)...);
}

template <typename scalar_t, typename... Args>
static void cu_ger(Args &&...args)
{
    static_assert(std::is_same_v<scalar_t, double> || std::is_same_v<scalar_t, float>,
                  "cu_ger implemented only for float and double.");

    if constexpr (std::is_same_v<scalar_t, double>)
        cublasDger(std::forward<Args>(args)...);
    else if constexpr (std::is_same_v<scalar_t, float>)
        cublasSger(std::forward<Args>(args)...);
}

template <typename scalar_t>
static void _apply_q_right_device(cublasHandle_t handle, int n, int m, int k, const scalar_t *h_A, int lda,
                                  const scalar_t *h_tau, scalar_t *d_C, int ldc)
{
    thrust::device_vector<scalar_t> v_dev(m);
    thrust::device_vector<scalar_t> temp_dev(n);

    scalar_t *d_temp = thrust::raw_pointer_cast(temp_dev.data());

    auto A = reshape(h_A, lda, k);

    for (int i = 0; i < k; ++i)
    {
        thrust::host_vector<scalar_t> v_host(m, 0.0);
        v_host[i] = 1.0;
        for (int j = i + 1; j < m; ++j)
            v_host[j] = A(j, i);

        v_dev = v_host;
        scalar_t *d_v = thrust::raw_pointer_cast(v_dev.data());

        // temp = C[:, :m] * v (matrix-vector multiply)
        scalar_t alpha = 1.0, beta = 0.0;
        cu_gemv<scalar_t>(handle, CUBLAS_OP_N, n, m, &alpha, d_C, ldc, d_v, 1, &beta, d_temp, 1);

        // C[:, :m] -= tau * temp * v' (rank-1 update)
        scalar_t tau_neg = -h_tau[i];
        cu_ger<scalar_t>(handle, n, m, &tau_neg, d_temp, 1, d_v, 1, d_C, ldc);
    }
}

void cuddh::apply_q_right_device(cublasHandle_t handle, int n, int m, int k, const double *h_A, int lda,
                                 const double *h_tau, double *d_C, int ldc)
{
    _apply_q_right_device<double>(handle, n, m, k, h_A, lda, h_tau, d_C, ldc);
}

void cuddh::apply_q_right_device(cublasHandle_t handle, int n, int m, int k, const float *h_A, int lda,
                                 const float *h_tau, float *d_C, int ldc)
{
    _apply_q_right_device<float>(handle, n, m, k, h_A, lda, h_tau, d_C, ldc);
}