#ifndef CUDDH_FACE_MASS_MATRIX_HPP
#define CUDDH_FACE_MASS_MATRIX_HPP

#include "H1Space2D.hpp"
#include "HostDeviceArray.hpp"
#include "Operator.hpp"
#include "forall.hpp"
#include "linalg.hpp"

namespace cuddh
{
    /// @brief m(u, phi) = (a(x) * u, phi) for all phi in a TraceSpace2D
    class FaceMassMatrix : public Operator
    {
    public:
        FaceMassMatrix(const TraceSpace2D &fs, const double *d_a = nullptr);

        /// @brief y[i] <- y[i] + c * (x, phi[i]),
        /// where phi[i] is the i-th basis function in the TraceSpace2D.
        /// @param c scalar coefficient
        /// @param x a vector in the TraceSpace2D
        /// @param y a vector in the TraceSpace2D. On exit, y[i] <- y[i] + c * (x, phi[i]).
        void action(double c, const double *x, double *y) const override;

        /// @brief y[i] = (x, phi[i])
        void action(const double *x, double *y) const override;

        /// @brief returns the (diagonal) mass matrix as VectorWrapper of managed memory
        VectorWrapper<const double> to_device() const
        {
            return reshape(_m, _m.size());
        }

    private:
        thrust::universal_vector<double> _m;
    };

    template <typename Func>
    thrust::universal_vector<double> trace(const TraceSpace2D &fs, const Func &f)
    {
        const int fdof = fs.size();

        auto x = fs.h1_space().physical_coordinates(MemorySpace::DEVICE);
        auto gI = fs.global_indices(MemorySpace::DEVICE);

        thrust::universal_vector<double> F(fdof);
        double *d_F = thrust::raw_pointer_cast(F.data());

        forall(fdof, [=] __device__(int i) {
            int gi = gI[i];
            double xi[] = {x(0, gi), x(1, gi)};
            d_F[i] = f(xi);
        });

        return F;
    }
} // namespace cuddh

#endif
