#pragma once

#include <cblas.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cmath>

#include "Operator.hpp"
#include "cuddh_config.hpp"

namespace cuddh
{
    /** @brief Device Linear Algebra namespace - contains operations on device memory */
    namespace dla
    {
        /** @brief y[i] <- a * x[i] + b * y[i] (device) */
        void axpby(int n, double a, const double *x, double b, double *y);
        void axpby(int n, float a, const float *x, float b, float *y);

        /** @brief the dot product between x and y (device) */
        double dot(int n, const double *x, const double *y);
        float dot(int n, const float *x, const float *y);

        /** @brief the euclidean norm of x: ||x|| (device) */
        inline double norm(int n, const double *x)
        {
            return std::sqrt(dla::dot(n, x, x));
        }
        inline float norm(int n, const float *x)
        {
            return std::sqrt(dla::dot(n, x, x));
        }

        /** @brief ||x - y|| (device) */
        double dist(int n, const double *x, const double *y);
        float dist(int n, const float *x, const float *y);

        /** @brief y[i] <- x[i] (device) */
        void copy(int n, const double *x, double *y);
        void copy(int n, const float *x, float *y);
        void copy(int n, const int *x, int *y);

        /** @brief x[i] <- a * x[i] (device) */
        void scal(int n, double a, double *x);
        void scal(int n, float a, float *x);

        /** @brief x[i] <- a (device) */
        void fill(int n, double a, double *x);
        void fill(int n, float a, float *x);
        void fill(int n, int a, int *x);

        /** @brief x[i] <- 0 (device) */
        inline void zeros(int n, double *x)
        {
            cudaMemset(x, 0, n * sizeof(double));
        }
        inline void zeros(int n, float *x)
        {
            cudaMemset(x, 0, n * sizeof(float));
        }
        inline void zeros(int n, int *x)
        {
            cudaMemset(x, 0, n * sizeof(int));
        }

        /** @brief x[i] <- 1 (device) */
        inline void ones(int n, double *x)
        {
            fill(n, 1.0, x);
        }
        inline void ones(int n, float *x)
        {
            fill(n, 1.0f, x);
        }
        inline void ones(int n, int *x)
        {
            cudaMemset(x, 1, n * sizeof(int));
        }

        bool is_symmetric(const Operator<float> &A, float tol = 1e-6f);
        bool is_symmetric(const Operator<double> &A, double tol = 1e-12);

        /**
         * @brief Performs matrix-matrix multiplication on the device: C = alpha * op(A) * op(B) + beta * C
         *
         * @param[in] handle cuBLAS handle
         * @param[in] m Number of rows of the matrix op(A) and of the matrix C
         * @param[in] n Number of columns of the matrix op(B) and of the matrix C
         * @param[in] k Number of columns of the matrix op(A) and rows of the matrix op(B)
         * @param[in] alpha Scalar multiplier for the product of matrices A and B
         * @param[in] d_A Pointer to the first element of matrix A on the device, stored in column-major order
         * @param[in] lda leading dimension of A (the number of elements between successive columns in memory)
         * @param[in] d_B Pointer to the first element of matrix B on the device, stored in column-major order
         * @param[in] ldb leading dimension of B (the number of elements between successive columns in memory)
         * @param[in] beta Scalar multiplier for the matrix C
         * @param[in,out] d_C Pointer to the first element of matrix C on the device, stored in column-major order
         * @param[in] ldc leading dimension of C (the number of elements between successive columns in memory)
         * @param[in] trans_A Whether to transpose matrix A
         * @param[in] trans_B Whether to transpose matrix B
         */
        inline void gemm(cublasHandle_t handle, int m, int n, int k, double alpha, const double *d_A, int lda,
                         const double *d_B, int ldb, double beta, double *d_C, int ldc, bool trans_A = false,
                         bool trans_B = false)
        {
            cublasOperation_t opA = trans_A ? CUBLAS_OP_T : CUBLAS_OP_N;
            cublasOperation_t opB = trans_B ? CUBLAS_OP_T : CUBLAS_OP_N;
            cublasDgemm(handle, opA, opB, m, n, k, &alpha, d_A, lda, d_B, ldb, &beta, d_C, ldc);
        }

        /**
         * @brief Performs matrix-matrix multiplication on the device: C = alpha * op(A) * op(B) + beta * C
         *
         * @param[in] handle cuBLAS handle
         * @param[in] m Number of rows of the matrix op(A) and of the matrix C
         * @param[in] n Number of columns of the matrix op(B) and of the matrix C
         * @param[in] k Number of columns of the matrix op(A) and rows of the matrix op(B)
         * @param[in] alpha Scalar multiplier for the product of matrices A and B
         * @param[in] d_A Pointer to the first element of matrix A on the device, stored in column-major order
         * @param[in] lda leading dimension of A (the number of elements between successive columns in memory)
         * @param[in] d_B Pointer to the first element of matrix B on the device, stored in column-major order
         * @param[in] ldb leading dimension of B (the number of elements between successive columns in memory)
         * @param[in] beta Scalar multiplier for the matrix C
         * @param[in,out] d_C Pointer to the first element of matrix C on the device, stored in column-major order
         * @param[in] ldc leading dimension of C (the number of elements between successive columns in memory)
         * @param[in] trans_A Whether to transpose matrix A
         * @param[in] trans_B Whether to transpose matrix B
         */
        inline void gemm(cublasHandle_t handle, int m, int n, int k, float alpha, const float *d_A, int lda,
                         const float *d_B, int ldb, float beta, float *d_C, int ldc, bool trans_A = false,
                         bool trans_B = false)
        {
            cublasOperation_t opA = trans_A ? CUBLAS_OP_T : CUBLAS_OP_N;
            cublasOperation_t opB = trans_B ? CUBLAS_OP_T : CUBLAS_OP_N;
            cublasSgemm(handle, opA, opB, m, n, k, &alpha, d_A, lda, d_B, ldb, &beta, d_C, ldc);
        }
    } // namespace dla

    /** @brief Host Linear Algebra namespace - contains operations on host memory */
    namespace hla
    {
        /**
         * @brief Performs matrix-matrix multiplication: C = alpha * op(A) * op(B) + beta * C (host)
         *
         * @param[in] m Number of rows of the matrix op(A) and of the matrix C
         * @param[in] n Number of columns of the matrix op(B) and of the matrix C
         * @param[in] k Number of columns of the matrix op(A) and rows of the matrix op(B)
         * @param[in] alpha Scalar multiplier for the product of matrices A and B
         * @param[in] A Pointer to the first element of matrix A, stored in column-major order
         * @param[in] lda leading dimension of A (the number of elements between successive columns in memory)
         * @param[in] B Pointer to the first element of matrix B, stored in column-major order
         * @param[in] ldb leading dimension of B (the number of elements between successive columns in memory)
         * @param[in] beta Scalar multiplier for the matrix C
         * @param[in,out] C Pointer to the first element of matrix C, stored in column-major order
         * @param[in] ldc leading dimension of C (the number of elements between successive columns in memory)
         * @param[in] trans_A Whether to transpose matrix A
         * @param[in] trans_B Whether to transpose matrix B
         */
        inline void gemm(int m, int n, int k, double alpha, const double *A, int lda, const double *B, int ldb,
                         double beta, double *C, int ldc, bool trans_A = false, bool trans_B = false)
        {
            cblas_dgemm(CblasColMajor, trans_A ? CblasTrans : CblasNoTrans, trans_B ? CblasTrans : CblasNoTrans, m, n,
                        k, alpha, A, lda, B, ldb, beta, C, ldc);
        }

        /**
         * @brief Performs matrix-matrix multiplication: C = alpha * op(A) * op(B) + beta * C (host)
         *
         * @param[in] m Number of rows of the matrix op(A) and of the matrix C
         * @param[in] n Number of columns of the matrix op(B) and of the matrix C
         * @param[in] k Number of columns of the matrix op(A) and rows of the matrix op(B)
         * @param[in] alpha Scalar multiplier for the product of matrices A and B
         * @param[in] A Pointer to the first element of matrix A, stored in column-major order
         * @param[in] lda leading dimension of A (the number of elements between successive columns in memory)
         * @param[in] B Pointer to the first element of matrix B, stored in column-major order
         * @param[in] ldb leading dimension of B (the number of elements between successive columns in memory)
         * @param[in] beta Scalar multiplier for the matrix C
         * @param[in,out] C Pointer to the first element of matrix C, stored in column-major order
         * @param[in] ldc leading dimension of C (the number of elements between successive columns in memory)
         * @param[in] trans_A Whether to transpose matrix A
         * @param[in] trans_B Whether to transpose matrix B
         */
        inline void gemm(int m, int n, int k, float alpha, const float *A, int lda, const float *B, int ldb, float beta,
                         float *C, int ldc, bool trans_A = false, bool trans_B = false)
        {
            cblas_sgemm(CblasColMajor, trans_A ? CblasTrans : CblasNoTrans, trans_B ? CblasTrans : CblasNoTrans, m, n,
                        k, alpha, A, lda, B, ldb, beta, C, ldc);
        }
    } // namespace hla

} // namespace cuddh
