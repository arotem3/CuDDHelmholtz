#pragma once

#include <assert.h>
#include <cuda_runtime.h>

#include <concepts>
#include <format>
#include <functional>
#include <type_traits>
#include <unordered_set>

#include "DDFaceMassMatrix3D.hpp"
#include "DDKernelConfig.hpp"
#include "DDMassMatrix3D.hpp"
#include "DDStiffnessMatrix3D.hpp"
#include "DDSymmetrize.hpp"
#include "DDWaveHoltz3D.hpp"
#include "EnsembleSpace3D.hpp"
#include "HostDeviceArray.hpp"
#include "LambdaDOFData.hpp"
#include "LinearSolvers/minres.hpp"
#include "Operator.hpp"
#include "Operators3D/MassMatrix3D.hpp"
#include "cuddh_config.hpp"
#include "cuddh_error.hpp"
#include "forall.hpp"
#include "linalg.hpp"

namespace cuddh
{
    template <typename scalar_t, SubdomainSolver Solver>
    struct DDSolverData3D
    {};

    template <typename scalar_t>
    struct DDSolverData3D<scalar_t, SubdomainSolver::WaveHoltz>
    {
        DDWaveHoltz<scalar_t> W;
        int waveholtz_iterations{-1};
    };

    template <typename scalar_t>
    struct DDSolverData3D<scalar_t, SubdomainSolver::MINRES>
    {
        DDMassMatrix3D<scalar_t> mass;
        DDFaceMassMatrix3D<scalar_t> face_mass;
        scalar_t omega{};
    };

    /**
     * @brief Operator for Helmholtz domain decomposition substructured problem in 3D.
     *
     * The scalar type scalar_t is either float or double and is the scalar type
     * in which the substructured problem is solved. The original finite element
     * problem is always in double precision.
     *
     * @tparam scalar_t   float or double
     * @tparam Solver     SubdomainSolver::WaveHoltz (default) or
     *                    SubdomainSolver::MINRES
     */
    template <std::floating_point scalar_t, SubdomainSolver Solver = SubdomainSolver::WaveHoltz>
    class DDSubstructuredOperator3D : public Operator<scalar_t>, private DDSolverData3D<scalar_t, Solver>
    {
    public:
        /// @brief initialize domain decomposition Helmholtz approximate solver.
        /// @param efem                 EnsembleSpace3D. Must be a uniform partition of the mesh into rectangular
        /// subdomains.
        /// @param omega                the Helmholtz frequency
        /// @param a                    the variable coefficient a(x)
        /// @param config               kernel launch configuration (TDOF). Block size is determined automatically from
        /// n_basis.
        /// @param waveholtz_iterations For WaveHoltz solver only: fixed number of iterations;
        ///                             -1 = residual-based stopping.
        DDSubstructuredOperator3D(const EnsembleSpace3D &efem, double omega, const GridFunc3D<double> &a,
                                  DDKernelConfig config = {}, int waveholtz_iterations = -1);

        ~DDSubstructuredOperator3D() = default;

        // Compute the right hand side `b` of the substructured problem from the
        // forcing `f` of the Helmholtz problem (i.e. the right hand side of the
        // finite element problem).
        void rhs(const double *f, scalar_t *b) const;

        // extract the finite element solution `u` from the solution of the
        // substructured problem `lambda` and the Helmholtz forcing `f`.
        void postprocess(const scalar_t *lambda, const double *f, double *u) const;

        /// @brief y <- F * x  (substructured operator action)
        void action(const scalar_t *x, scalar_t *y) const override;

        void action(scalar_t, const scalar_t *, scalar_t *) const override
        {
            cuddh_verify(false, printf("DDSubstructuredOperator3D::action(c, x, y) not implemented\n"));
        }

        /// @brief Returns a string describing the kernel launch configuration: threads per block and DOFs per thread.
        std::string kernel_str() const
        {
            return std::format("block_size = {} threads/block, tdof = {} DOFs/thread",
                               static_cast<int>(kernel_config.block_size), kernel_config.tdof);
        }

    private:
        void action(const double *fem_in, double *fem_out, const scalar_t *lambda_in, scalar_t *lambda_out) const;

    private:
        const EnsembleSpace3D &efem;

        int g_ndof;
        int g_elem;
        int n_basis;
        int n_domains;
        int n_lambda;
        int mx_dof;
        int mx_fdof;
        int mx_elem_per_dom;

        DDKernelConfig kernel_config;

        thrust::device_vector<LambdaDOFData<scalar_t>> _B;

        DDStiffnessMatrix3D<scalar_t> S;

        thrust::device_vector<scalar_t> _partition_of_unity;
        mutable thrust::device_vector<scalar_t> _work;
    };

    extern template class DDSubstructuredOperator3D<float>;
    extern template class DDSubstructuredOperator3D<double>;
    extern template class DDSubstructuredOperator3D<float, SubdomainSolver::MINRES>;
    extern template class DDSubstructuredOperator3D<double, SubdomainSolver::MINRES>;

    /**
     * @brief Domain decomposition Helmholtz solver in 3D.
     *
     * @tparam scalar_t  float or double
     * @tparam Solver    SubdomainSolver::WaveHoltz (default) or MINRES
     */
    template <std::floating_point scalar_t, SubdomainSolver InnerSolver = SubdomainSolver::WaveHoltz>
    class DDH3D : Solver<double>
    {
    public:
        DDH3D(const EnsembleSpace3D &efem, double omega, const GridFunc3D<double> &a, DDKernelConfig kernel_config = {},
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

        const DDSubstructuredOperator3D<scalar_t, InnerSolver> &op() const { return F; }

    private:
        DDSubstructuredOperator3D<scalar_t, InnerSolver> F;
        MINRES<scalar_t> solver;
        mutable thrust::device_vector<scalar_t> lambda;
        mutable thrust::device_vector<scalar_t> Y;
    };

    extern template class DDH3D<float>;
    extern template class DDH3D<double>;
    extern template class DDH3D<float, SubdomainSolver::MINRES>;
    extern template class DDH3D<double, SubdomainSolver::MINRES>;
} // namespace cuddh
