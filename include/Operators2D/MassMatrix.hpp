#pragma once

#include <thrust/device_vector.h>

#include "FEM2D/GridFunc2D.hpp"
#include "FEM2D/H1Space2D.hpp"
#include "Operator.hpp"
#include "SparseMatrix.hpp"
#include "forall.hpp"

namespace cuddh
{
    /// @brief m(u, v) = (u, v) or m(u, v) = (a(x)*u, v)
    class MassMatrix : public Operator<double>
    {
    public:
        /**
         * @brief Construct a new Mass Matrix m(u, v) = (a(x) * u, v)
         *
         * @param fem
         * @param a variable coefficient
         */
        MassMatrix(const H1Space2D &fem, const GridFunc2D<double> &a);
        MassMatrix(const H1Space2D &fem);

        /// @brief y <- y + c * M*x, where M is the mass matrix
        void action(double c, const double *x, double *y) const override;

        void action(const double *x, double *y) const override;

        /// @brief S <- S + c * M  (real sparse matrix)
        bool assemble(double c, SparseMatrix<double> &S) const override;

        /// @brief S <- S + c * M  (complex sparse matrix; c may be complex)
        bool assemble(std::complex<double> c, SparseMatrix<double, true> &S) const override;

        // returns the mass matrix as a dvec_wrapper of device memory
        VectorWrapper<const double> to_device() const
        {
            return reshape(thrust::raw_pointer_cast(_m.data()), _m.size());
        }

    private:
        const H1Space2D &fem;
        thrust::device_vector<double> _m;

        template <typename Func>
        friend void l2_project(double *d_F, const MassMatrix &M, const Func &f);
    };

    template <typename Func>
    void l2_project(double *d_F, const MassMatrix &M, const Func &f)
    {
        const int ndof = M.fem.size();

        auto x = M.fem.physical_coordinates(MemorySpace::DEVICE);
        auto m = M.to_device();

        forall(ndof, [=] __device__(int i) {
            double2 xi = x(i);
            d_F[i] = f(xi) * m[i];
        });
    }
} // namespace cuddh
