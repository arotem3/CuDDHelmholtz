#pragma once

#include <assert.h>
#include <cuda_runtime.h>

#include <cub/cub.cuh>
#include <cuda/std/array>
#include <format>
#include <functional>
#include <random>
#include <type_traits>
#include <unordered_set>

#include "DD2D.hpp"
#include "DDFaceMassMatrix.hpp"
#include "DDKernelConfig.hpp"
#include "DDMassMatrix.hpp"
#include "DDStiffnessMatrix.hpp"
#include "DDSymmetrize.hpp"
#include "DDWaveHoltz2D.hpp"
#include "EnsembleSpace.hpp"
#include "FEM2D/GridFunc2D.hpp"
#include "HostDeviceArray.hpp"
#include "LambdaDOFData.hpp"
#include "LinearSolvers/minres.hpp"
#include "Operator.hpp"
#include "Operators2D/MassMatrix.hpp"
#include "cuddh_config.hpp"
#include "cuddh_error.hpp"
#include "forall.hpp"
#include "linalg.hpp"

namespace cuddh
{
    template <typename scalar_t, SubdomainSolver Solver>
    struct DDSolverData
    {};

    template <typename scalar_t>
    struct DDSolverData<scalar_t, SubdomainSolver::WaveHoltz>
    {
        DDWaveHoltz<scalar_t> W;
        int waveholtz_iterations{-1};
    };

    template <typename scalar_t>
    struct DDSolverData<scalar_t, SubdomainSolver::MINRES>
    {
        DDMassMatrix<scalar_t> mass;
        DDFaceMassMatrix<scalar_t> face_mass;
        scalar_t omega{};
    };

    /**
     * @brief Operator for Helmholtz domain decomposition substructured problem.
     *
     * The scalar type scalar_t is either float or double and is the scalar type
     * in which the substructured problem is solved. The original finite element
     * problem is always in double precision.
     *
     * @tparam scalar_t   float or double
     * @tparam Solver     SubdomainSolver::WaveHoltz (default) or
     *                    SubdomainSolver::MINRES
     */
    template <typename scalar_t, SubdomainSolver Solver = SubdomainSolver::WaveHoltz>
    class DDSubstructuredOperator : public Operator<scalar_t>, private DDSolverData<scalar_t, Solver>
    {
        static_assert(std::is_same_v<scalar_t, float> || std::is_same_v<scalar_t, double>,
                      "scalar_t must be float or double");

    public:
        /// @brief Initialize domain decomposition Helmholtz approximate solver.
        /// @param omega                Helmholtz frequency
        /// @param a                    Variable coefficient a(x)
        /// @param fem                  H1Space2D on a uniform_rect mesh
        /// @param efem                 EnsembleSpace: uniform rectangular partition
        /// @param config               Kernel launch configuration
        /// @param waveholtz_iterations For WaveHoltz solver only: fixed number of
        ///                             iterations; -1 = residual-based stopping.
        DDSubstructuredOperator(const EnsembleSpace &efem, double omega, const GridFunc2D<double> &a,
                                DDKernelConfig config = {}, int waveholtz_iterations = -1);

        ~DDSubstructuredOperator() = default;

        /// Compute the right-hand side b of the substructured problem from the
        /// Helmholtz forcing f.
        void rhs(const double *f, scalar_t *b) const;

        /// Extract the FEM solution u from the substructured solution lambda and
        /// the Helmholtz forcing f.
        void postprocess(const scalar_t *lambda, const double *f, double *u) const;

        /// @brief y <- F * x  (substructured operator action)
        void action(const scalar_t *x, scalar_t *y) const override;

        void action(scalar_t, const scalar_t *, scalar_t *) const override
        {
            cuddh_verify(false, printf("DDSubstructuredOperator::action(c, x, y) not implemented\n"));
        }

        /// Returns a string describing the kernel launch configuration.
        std::string kernel_str() const
        {
            return std::format("block_size = {} threads/block, tdof = {} DOFs/thread",
                               static_cast<int>(kernel_config.block_size), kernel_config.tdof);
        }

    private:
        void action(const double *fem_in, double *fem_out, const scalar_t *lambda_in, scalar_t *lambda_out) const;

    private:
        DDKernelConfig kernel_config;

        int g_ndof;
        int g_elem;
        int n_basis;
        int n_domains;
        int n_lambda;
        int mx_dof;
        int mx_fdof;
        int mx_elem_per_dom;

        const EnsembleSpace &efem;

        thrust::device_vector<LambdaDOFData<scalar_t>> _B;

        DDStiffnessMatrix<scalar_t> S;

        thrust::device_vector<scalar_t> _partition_of_unity;
        mutable thrust::device_vector<scalar_t> _work;
    };

    extern template class DDSubstructuredOperator<float>;
    extern template class DDSubstructuredOperator<double>;
    extern template class DDSubstructuredOperator<float, SubdomainSolver::MINRES>;
    extern template class DDSubstructuredOperator<double, SubdomainSolver::MINRES>;

    /**
     * @brief Domain decomposition Helmholtz solver.
     *
     * @tparam scalar_t  float or double
     * @tparam Solver    SubdomainSolver::WaveHoltz (default) or MINRES
     */
    template <typename scalar_t, SubdomainSolver InnerSolver = SubdomainSolver::WaveHoltz>
    class DDH : public Solver<double>
    {
    public:
        DDH(const EnsembleSpace &efem, double omega, const GridFunc2D<double> &a, DDKernelConfig kernel_config = {},
            int waveholtz_iterations = -1)
            : Solver<double>(2 * efem.h1_space().size()),
              F(efem, omega, a, kernel_config, waveholtz_iterations),
              solver(F),
              lambda(F.ndof()),
              Y(F.ndof())
        {}

        SolverResults solve(double *x, const double *b, SolverParams opts = {}) const override
        {
            thrust::fill(lambda.begin(), lambda.end(), scalar_t(0));

            scalar_t *d_L = thrust::raw_pointer_cast(lambda.data());
            scalar_t *d_Y = thrust::raw_pointer_cast(Y.data());

            F.rhs(b, d_Y);
            SolverResults out = solver.solve(d_L, d_Y, opts);

            F.postprocess(d_L, b, x);

            return out;
        }

        const DDSubstructuredOperator<scalar_t, InnerSolver> &op() const { return F; }

    private:
        DDSubstructuredOperator<scalar_t, InnerSolver> F;
        MINRES<scalar_t> solver;

        mutable thrust::device_vector<scalar_t> lambda;
        mutable thrust::device_vector<scalar_t> Y;
    };

    extern template class DDH<float>;
    extern template class DDH<double>;
    extern template class DDH<float, SubdomainSolver::MINRES>;
    extern template class DDH<double, SubdomainSolver::MINRES>;
} // namespace cuddh
