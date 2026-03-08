#pragma once
#include <cblas.h>
#include <cublas_v2.h>
#include <lapacke.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <algorithm>
#include <limits>
#include <numeric>
#include <vector>

#include "Tensor.hpp"
#include "cuddh_config.hpp"
#include "cuddh_error.hpp"
#include "forall.hpp"
#include "linalg.hpp"

namespace cuddh
{
    template <typename real_t>
    constexpr void apply_givens(real_t &x, real_t &y, real_t c, real_t s)
    {
        real_t t = c * x + s * y;
        y = -s * x + c * y;
        x = t;
    }

    template <typename real_t>
    constexpr std::pair<real_t, real_t> compute_givens(real_t x, real_t y)
    {
        real_t r = std::hypot(x, y);
        if (r == 0)
            return {real_t(1), real_t(0)};
        return {x / r, y / r};
    }

    // Computes [X Y] <- [X Y] * [c -s; s c] for device vectors X and Y
    void rotate_vecs(int n, double *d_X, double *d_Y, double cs, double sn);
    void rotate_vecs(int n, float *d_X, float *d_Y, float cs, float sn);

    /**
     * @brief Solves an upper triangular system R * x = b
     *
     * @param ldr Leading dimension of R
     * @param n Size of the system
     * @param R Upper triangular matrix
     * @param b Right-hand side vector, overwritten with the solution
     */
    inline void solve_triu(int n, const double *R, int ldr, double *b)
    {
        int info = LAPACKE_dtrtrs(LAPACK_COL_MAJOR, 'U', 'N', 'N', n, 1, R, ldr, b, n);
        cuddh_verify(info == 0, {
            if (info < 0)
                printf("LAPACKE_dtrtrs: invalid argument index %d\n", -info);
            else
                printf("LAPACKE_dtrtrs: singular matrix; zero diagonal at index %d\n", info);
        });
    }

    /**
     * @brief Solves an upper triangular system R * x = b
     *
     * @param ldr Leading dimension of R
     * @param n Size of the system
     * @param R Upper triangular matrix
     * @param b Right-hand side vector, overwritten with the solution
     */
    inline void solve_triu(int n, const float *R, int ldr, float *b)
    {
        int info = LAPACKE_strtrs(LAPACK_COL_MAJOR, 'U', 'N', 'N', n, 1, R, ldr, b, n);
        cuddh_verify(info == 0, {
            if (info < 0)
                printf("LAPACKE_strtrs: invalid argument index %d\n", -info);
            else
                printf("LAPACKE_strtrs: singular matrix; zero diagonal at index %d\n", info);
        });
    }

    /**
     * @brief Performs right-side triangular matrix-matrix multiplication on the device: B = alpha * B * A
     *
     * @param[in] handle cuBLAS handle
     * @param[in] n Number of rows of the matrix B
     * @param[in] k Number of columns of the matrix B and size of the triangular matrix A
     * @param[in] alpha Scalar multiplier for the product of matrices B and A
     * @param[in] h_A Pointer to the first element of the triangular matrix A on the host, stored in column-major order
     * @param[in] lda leading dimension of A (the number of elements between successive columns in memory)
     * @param[in,out] d_B Pointer to the first element of matrix B on the device, stored in column-major order
     * @param[in] ldb leading dimension of B (the number of elements between successive columns in memory)
     */
    inline void trmm_right_device(cublasHandle_t handle, int n, int k, double alpha, const double *h_A, int lda,
                                  double *d_B, int ldb)
    {
        thrust::device_vector<double> A_dev(lda * k);
        cudaMemcpy(thrust::raw_pointer_cast(A_dev.data()), h_A, lda * k * sizeof(double), cudaMemcpyHostToDevice);
        const double *d_A = thrust::raw_pointer_cast(A_dev.data());
        cublasDtrmm(handle, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, n, k, &alpha,
                    d_A, lda, d_B, ldb, d_B, ldb);
    }

    /**
     * @brief Performs right-side triangular matrix-matrix multiplication on the device: B = alpha * B * A
     *
     * @param[in] handle cuBLAS handle
     * @param[in] n Number of rows of the matrix B
     * @param[in] k Number of columns of the matrix B and size of the triangular matrix A
     * @param[in] alpha Scalar multiplier for the product of matrices B and A
     * @param[in] h_A Pointer to the first element of the triangular matrix A on the host, stored in column-major order
     * @param[in] lda leading dimension of A (the number of elements between successive columns in memory)
     * @param[in,out] d_B Pointer to the first element of matrix B on the device, stored in column-major order
     * @param[in] ldb leading dimension of B (the number of elements between successive columns in memory)
     */
    inline void trmm_right_device(cublasHandle_t handle, int n, int k, float alpha, const float *h_A, int lda,
                                  float *d_B, int ldb)
    {
        thrust::device_vector<float> A_dev(lda * k);
        cudaMemcpy(thrust::raw_pointer_cast(A_dev.data()), h_A, lda * k * sizeof(float), cudaMemcpyHostToDevice);
        const float *d_A = thrust::raw_pointer_cast(A_dev.data());
        cublasStrmm(handle, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, n, k, &alpha,
                    d_A, lda, d_B, ldb, d_B, ldb);
    }

    /**
     * @brief Computes the invariant subspace of a real generalized Schur form.
     *
     * @param[in] n The order of the matrices A and B
     * @param[in] nkeep The number of eigenvalues to keep
     * @param[in,out] A Pointer to the first element of matrix A, stored in column-major order. On exit, A is
     * overwritten with the AA matrix from the QZ decomposition.
     * @param[in] lda leading dimension of A (the number of elements between successive columns in memory)
     * @param[in,out] B Pointer to the first element of matrix B, stored in column-major order. On exit, B is
     * overwritten with the BB matrix from the QZ decomposition.
     * @param[in] ldb leading dimension of B (the number of elements between successive columns in memory)
     * @return Matrix<double> The selected invariant subspace vectors
     */
    Matrix<double> qz_invariant_space(int n, int nkeep, double *A, int lda, double *B, int ldb);
    Matrix<float> qz_invariant_space(int n, int nkeep, float *A, int lda, float *B, int ldb);

    /**
     * @brief Thin QR decomposition of an m x n matrix A (m >= n) using LAPACKE. The Householder reflectors are stored
     * in the lower triangle of A and the scalar factors in tau. The upper triangle of A contains the R factor. This is
     * a thin QR decomposition, so only the first n columns of A are used and the output Q is represented implicitly by
     * the reflectors and tau. The caller can use the reflectors and tau to apply Q to other matrices without explicitly
     * forming Q, which saves memory and computation when m >> n.
     *
     * @param[in] m rows of A
     * @param[in] n columns of A
     * @param[in,out] A input matrix (m x n) stored in column-major order. On exit, the upper triangle contains R and
     * the lower triangle contains the Householder vectors.
     * @param[in] lda leading dimension of A
     * @param[out] tau scalar factors of the elementary reflectors
     */
    inline void qr_decomp(int m, int n, double *A, int lda, std::vector<double> &tau)
    {
        tau.resize(std::min(m, n));
        lapack_int info = LAPACKE_dgeqrf(LAPACK_COL_MAJOR, m, n, A, lda, tau.data());
        cuddh_verify(info == 0, {
            if (info < 0)
                printf("LAPACKE_dgeqrf: invalid argument index %d\n", -info);
            else
                printf("LAPACKE_dgeqrf: failed with info=%d\n", info);
        });
    }

    /**
     * @brief Thin QR decomposition of an m x n matrix A (m >= n) using LAPACKE. The Householder reflectors are stored
     * in the lower triangle of A and the scalar factors in tau. The upper triangle of A contains the R factor. This is
     * a thin QR decomposition, so only the first n columns of A are used and the output Q is represented implicitly by
     * the reflectors and tau. The caller can use the reflectors and tau to apply Q to other matrices without explicitly
     * forming Q, which saves memory and computation when m >> n.
     *
     * @param[in] m rows of A
     * @param[in] n columns of A
     * @param[in,out] A input matrix (m x n) stored in column-major order. On exit, the upper triangle contains R and
     * the lower triangle contains the Householder vectors.
     * @param[in] lda leading dimension of A
     * @param[out] tau scalar factors of the elementary reflectors
     */
    inline void qr_decomp(int m, int n, float *A, int lda, std::vector<float> &tau)
    {
        tau.resize(std::min(m, n));
        lapack_int info = LAPACKE_sgeqrf(LAPACK_COL_MAJOR, m, n, A, lda, tau.data());
        cuddh_verify(info == 0, {
            if (info < 0)
                printf("LAPACKE_sgeqrf: invalid argument index %d\n", -info);
            else
                printf("LAPACKE_sgeqrf: failed with info=%d\n", info);
        });
    }

    /**
     * @brief Triangular solve with multiple right-hand sides: B = alpha * B * A^{-1} where A is upper triangular. This
     * is used to apply the inverse of the R factor from QR to the Krylov basis.
     *
     * @param m number of rows of B
     * @param n number of columns of B
     * @param alpha scalar multiplier for the operation
     * @param A upper triangular matrix (n x n) stored in column-major order
     * @param lda leading dimension of A
     * @param B matrix (m x n) stored in column-major order, overwritten with the result
     * @param ldb leading dimension of B
     */
    inline void trsm_upper_right(int m, int n, double alpha, const double *A, int lda, double *B, int ldb)
    {
        cblas_dtrsm(CblasColMajor, CblasRight, CblasUpper, CblasNoTrans, CblasNonUnit, m, n, alpha, A, lda, B, ldb);
    }

    /**
     * @brief Triangular solve with multiple right-hand sides: B = alpha * B * A^{-1} where A is upper triangular. This
     * is used to apply the inverse of the R factor from QR to the Krylov basis.
     *
     * @param m number of rows of B
     * @param n number of columns of B
     * @param alpha scalar multiplier for the operation
     * @param A upper triangular matrix (n x n) stored in column-major order
     * @param lda leading dimension of A
     * @param B matrix (m x n) stored in column-major order, overwritten with the result
     * @param ldb leading dimension of B
     */
    inline void trsm_upper_right(int m, int n, float alpha, const float *A, int lda, float *B, int ldb)
    {
        cblas_strsm(CblasColMajor, CblasRight, CblasUpper, CblasNoTrans, CblasNonUnit, m, n, alpha, A, lda, B, ldb);
    }

    /**
     * @brief Applies the orthogonal matrix Q from a thin QR decomposition to a matrix C on the right: C <- C * Q.
     *
     * @details The matrix Q is represented implicitly by the Householder reflectors stored in the lower triangle of A
     * and the scalar factors in tau. This is used to apply the Q factor from QR to the Krylov basis without explicitly
     * forming Q, which saves memory and computation when m >> n. The first k columns of A and the first k elements of
     * tau are used, where k is the number of Householder reflectors (which is typically equal to n, the number of
     * columns of A). The operation is performed on the device using cuBLAS, so the input matrix C is on the device and
     * the Householder vectors and tau are copied from the host to the device as needed.
     *
     * @param handle cuBLAS handle
     * @param n number of rows of C
     * @param m number of columns of C
     * @param k number of Householder reflectors (typically equal to the number of columns of A)
     * @param h_A pointer to the first element of matrix A on the host, stored in column-major order. The Householder
     * vectors are stored in the lower triangle of A.
     * @param lda leading dimension of A (the number of elements between successive columns in memory)
     * @param h_tau pointer to the scalar factors of the Householder reflectors on the host
     * @param d_C pointer to the first element of matrix C on the device, stored in column-major order. On exit, C is
     * overwritten with C * Q.
     * @param ldc leading dimension of C (the number of elements between successive columns in memory)
     */
    void apply_q_right_device(cublasHandle_t handle, int n, int m, int k, const double *h_A, int lda,
                              const double *h_tau, double *d_C, int ldc);
    void apply_q_right_device(cublasHandle_t handle, int n, int m, int k, const float *h_A, int lda, const float *h_tau,
                              float *d_C, int ldc);
} // namespace cuddh
