#pragma once
#include "LinearSolvers/Arnoldi.hpp"
#include "LinearSolvers/KrylovHelpers.hpp"
#include "LinearSolvers/SolverBase.hpp"

namespace cuddh
{
    /**
     * @brief Flexible GCRO-DR solver.
     *
     * @tparam real_t
     */
    template <typename real_t>
    class GCRO : private BaseArnoldiSolver<real_t>
    {
    public:
        GCRO(int n, Operator<real_t> &A, Operator<real_t> *M = nullptr, int kdim = 40, int edim = 20);

        ~GCRO() { cublasDestroy(cublas_handle); }

        constexpr void toggle_deflation_update(bool update) { update_deflation = update; }

        constexpr void reset_deflation() { active_edim = 0; }

        SolverResults solve(real_t *x, const real_t *b, SolverParams opts = {}) const;

    private:
        // apply deflation correction
        void deflate(real_t *x, real_t *r) const;

        // Update W and Z with new deflation vectors. m is the Krylov dimension of the last cycle
        void compute_deflation_ritz_vecs(int m) const;

    private:
        using BaseArnoldiSolver<real_t>::kdim;
        using BaseArnoldiSolver<real_t>::n;
        using BaseArnoldiSolver<real_t>::_W;
        using BaseArnoldiSolver<real_t>::_Z;
        using BaseArnoldiSolver<real_t>::H;

        const int edim;          // target deflation dimension
        mutable int active_edim; // current deflation dimension

        bool update_deflation;

        mutable thrust::device_vector<real_t> _r; // residual vector, (n)

        cublasHandle_t cublas_handle; // Persistent cuBLAS handle
    };

    extern template class GCRO<float>;
    extern template class GCRO<double>;
} // namespace cuddh
