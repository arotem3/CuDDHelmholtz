#pragma once

#include <type_traits>

#include "FEM2D/H1Space2D.hpp"
#include "FEM2D/TraceFunc2D.hpp"
#include "FEM3D/H1Space3D.hpp"
#include "FEM3D/TraceFunc3D.hpp"
#include "HostDeviceArray.hpp"
#include "Operator.hpp"
#include "forall.hpp"
#include "linalg.hpp"

namespace cuddh
{
    /**
     * @brief Wrap an operator so output is projected to the homogeneous trace subspace.
     *
     * This corresponds to solving in H1_0 by applying TraceSpace::orth after each
     * operator action.
     */
    template <typename TraceSpaceT>
    class HomogeneousTraceOperator : public Operator<double>
    {
    public:
        HomogeneousTraceOperator(const Operator<double> &A, const TraceSpaceT &trace)
            : Operator<double>(A.ndof()), _A(A), _trace(trace)
        {}

        void action(double c, const double *x, double *y) const override
        {
            _A.action(c, x, y);
            _trace.orth(y);
        }

        void action(const double *x, double *y) const override
        {
            _A.action(x, y);
            _trace.orth(y);
        }

    private:
        const Operator<double> &_A;
        const TraceSpaceT &_trace;
    };

    /**
     * @brief Generic Dirichlet elimination helper.
     *
     * Public workflow:
     *   1) set(...)
     *   2) apply_rhs(A, rhs)
     *   3) solve homogeneous problem
     *   4) recover_solution(u)
     */
    template <typename TraceSpaceT>
    class DirichletBoundaryCondition
    {
    public:
        explicit DirichletBoundaryCondition(const TraceSpaceT &trace)
            : _trace(trace), _q(trace.size()), _G(trace.h1_space().size())
        {
            dla::zeros(_trace.size(), _q.device_write());
            dla::zeros(_trace.h1_space().size(), _G.device_write());
        }

        template <typename BoundaryFunc>
        DirichletBoundaryCondition(const TraceSpaceT &trace, const BoundaryFunc &g) : DirichletBoundaryCondition(trace)
        {
            set(g);
        }

        /// @brief Set Dirichlet data by evaluating a boundary function on trace DOFs.
        template <typename BoundaryFunc>
        void set(const BoundaryFunc &g)
        {
            auto x = _trace.h1_space().physical_coordinates(MemorySpace::DEVICE);
            auto global = _trace.global_indices(MemorySpace::DEVICE);
            double *q = _q.device_write();

            const int n = _trace.size();
            forall(n, [=] __device__(int i) { q[i] = g(x[global[i]]); });

            rebuild_extension();
        }

        /// @brief Set Dirichlet data from TraceFunc2D.
        template <typename value_t>
        void set(const TraceFunc2D<value_t> &tf)
            requires(std::is_same_v<TraceSpaceT, TraceSpace2D>)
        {
            cuddh_verify(
                &tf.trace_space() == &_trace,
                printf("DirichletBoundaryCondition::set error: TraceFunc2D belongs to a different trace space.\n"));

            const int n_basis = _trace.h1_space().basis().size();
            const int n_faces = _trace.n_faces();
            auto T = tf.read(MemorySpace::DEVICE);
            auto I = _trace.subspace_indices(MemorySpace::DEVICE);
            auto K = _trace.global_indices(MemorySpace::DEVICE);
            double *q = _q.device_write();

            forall_1d(n_basis, n_faces, [=] __device__(int f) mutable {
                const int i = threadIdx.x;
                q[I(i, f)] = static_cast<double>(T(i, f));
            });

            rebuild_extension();
        }

        /// @brief Set Dirichlet data from TraceFunc3D.
        template <typename value_t>
        void set(const TraceFunc3D<value_t> &tf)
            requires(std::is_same_v<TraceSpaceT, TraceSpace3D>)
        {
            cuddh_verify(
                &tf.trace_space() == &_trace,
                printf("DirichletBoundaryCondition::set error: TraceFunc3D belongs to a different trace space.\n"));

            const int n_basis = _trace.h1_space().basis().size();
            const int n_faces = _trace.n_faces();
            auto T = tf.read(MemorySpace::DEVICE);
            auto I = _trace.subspace_indices(MemorySpace::DEVICE);
            double *q = _q.device_write();

            forall_2d(n_basis, n_basis, n_faces, [=] __device__(int f) mutable {
                const int i = threadIdx.x;
                const int j = threadIdx.y;
                q[I(i, j, f)] = static_cast<double>(T(i, j, f));
            });

            rebuild_extension();
        }

        /// @brief Apply elimination to RHS: rhs <- orth(rhs) - A * E(q).
        template <typename OpT>
        void apply_rhs(const OpT &A, double *rhs) const
        {
            A.action(-1.0, _G.device_read(), rhs);
            _trace.orth(rhs);
        }

        /// @brief Recover full solution: u <- u + E(q).
        void recover_solution(double *u) const { dla::axpby(_trace.h1_space().size(), 1.0, _G.device_read(), 1.0, u); }

    private:
        void rebuild_extension()
        {
            double *G = _G.device_write();
            dla::zeros(_trace.h1_space().size(), G);
            _trace.prolong(_q.device_read(), G);
        }

        const TraceSpaceT &_trace;
        host_device_dvec _q;
        host_device_dvec _G;
    };

    using DirichletBC2D = DirichletBoundaryCondition<TraceSpace2D>;
    using DirichletBC3D = DirichletBoundaryCondition<TraceSpace3D>;

    using HomogeneousDirichletOperator2D = HomogeneousTraceOperator<TraceSpace2D>;
    using HomogeneousDirichletOperator3D = HomogeneousTraceOperator<TraceSpace3D>;
} // namespace cuddh