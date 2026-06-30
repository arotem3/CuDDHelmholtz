#pragma once

#include "FEM3D/GridFunc3D.hpp"
#include "FEM3D/H1Space3D.hpp"
#include "HostDeviceArray.hpp"
#include "Operator.hpp"
#include "SparseMatrix.hpp"
#include "Tensor.hpp"
#include "forall.hpp"

namespace cuddh
{
    /**
     * @brief Boundary mass matrix: m(u, phi) = <a(x) * u, phi> for phi in H1Space3D.
     *
     * Stores a diagonal of size fem.size(). Interior DOF entries are zero;
     * boundary DOF entries accumulate face quadrature contributions.
     * This matches the 2D FaceMassMatrix interface.
     */
    class FaceMassMatrix3D : public Operator<double>
    {
    public:
        /// @brief Unweighted: m(u, phi) = <u, phi>
        FaceMassMatrix3D(const TraceSpace3D &tr);

        /// @brief Weighted: m(u, phi) = <a(x) u, phi>
        FaceMassMatrix3D(const TraceSpace3D &tr, const GridFunc3D<double> &a);

        /// @brief y[i] <- y[i] + c * <x, phi[i]>  (H1Space3D-sized vectors)
        void action(double c, const double *x, double *y) const override;

        /// @brief y[i] = <x, phi[i]>  (H1Space3D-sized vectors)
        void action(const double *x, double *y) const override;

        /// @brief S <- S + c * H  (real sparse matrix)
        bool assemble(double c, SparseMatrix<double> &S) const override;

        /// @brief S <- S + c * H  (complex sparse matrix; c may be complex)
        bool assemble(std::complex<double> c, SparseMatrix<double, true> &S) const override;

        /// @brief Returns the diagonal mass vector (length fem.size()) on the device.
        const_dvec_wrapper to_device() const { return reshape(_m.device_read(), _m.size()); }

    private:
        const H1Space3D &_fem;
        HostDeviceArray<double> _m;
    };
} // namespace cuddh
