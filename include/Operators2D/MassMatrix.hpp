#ifndef CUDDH_MASS_MATRIX_HPP
#define CUDDH_MASS_MATRIX_HPP

#include "H1Space2D.hpp"
#include "HostDeviceArray.hpp"
#include "forall.hpp"
#include "linalg.hpp"

namespace cuddh
{
    /// @brief m(u, v) = (u, v) or m(u, v) = (a(x)*u, v)
    class MassMatrix : public Operator
    {
    public:
        /// @brief initialize weighted mass matrix m(u, v) = (a(x)*u, v)
        /// @param a DEVICE. H1Space2D vector representing the function a.
        /// @param fem
        MassMatrix(const H1Space2D &fem, const double *a = nullptr);

        /// @brief y <- y + c * M*x, where M is the mass matrix
        void action(double c, const double *x, double *y) const override;

        void action(const double *x, double *y) const override;

        // returns the mass matrix as a dvec_wrapper of managed memory
        VectorWrapper<const double> to_device() const
        {
            return reshape(_m, _m.size());
        }

    private:
        const H1Space2D &fem;
        thrust::universal_vector<double> _m;

        template <typename Func>
        friend thrust::universal_vector<double> l2_project(const MassMatrix &M, Func &&f);
    };

    template <typename Func>
    thrust::universal_vector<double> l2_project(const MassMatrix &M, Func &&f)
    {
        const int ndof = M.fem.size();
        thrust::universal_vector<double> F(ndof, 0.0);
        double *d_F = thrust::raw_pointer_cast(F.data());

        auto x = M.fem.physical_coordinates(MemorySpace::DEVICE);
        auto m = M.to_device();

        forall(ndof, [=] __device__(int i) {
            double xi[] = {x(0, i), x(1, i)};
            d_F[i] = f(xi) * m[i];
        });

        return F;
    }
} // namespace cuddh

#endif