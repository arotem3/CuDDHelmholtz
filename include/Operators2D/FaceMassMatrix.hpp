#pragma once

#include "FEM2D/GridFunc2D.hpp"
#include "FEM2D/H1Space2D.hpp"
#include "FEM2D/TraceFunc2D.hpp"
#include "HostDeviceArray.hpp"
#include "Operator.hpp"
#include "forall.hpp"
#include "linalg.hpp"

namespace cuddh
{
    /// @brief m(u, phi) = <a(x) * u, phi>
    class FaceMassMatrix : public Operator<double>
    {
    public:
        FaceMassMatrix(const TraceSpace2D &fs, const GridFunc2D<double> &a);
        FaceMassMatrix(const TraceSpace2D &fs);

        /// @brief y[i] <- y[i] + c * <x, phi[i]>
        /// @param c scalar coefficient
        /// @param x a vector in the H1Space2D
        /// @param y a vector in the H1Space2D. On exit, y[i] <- y[i] + c * <x, phi[i]>
        void action(double c, const double *x, double *y) const override;

        /// @brief y[i] = <x, phi[i]>
        void action(const double *x, double *y) const override;

        /// @brief returns the (diagonal) mass matrix as VectorWrapper of managed memory
        VectorWrapper<const double> to_device() const
        {
            return reshape(thrust::raw_pointer_cast(_m.data()), _m.size());
        }

    private:
        thrust::device_vector<double> _m;
    };
} // namespace cuddh
