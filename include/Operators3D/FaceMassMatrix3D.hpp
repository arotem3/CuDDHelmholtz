#pragma once

#include "FEM3D/GridFunc3D.hpp"
#include "FEM3D/H1Space3D.hpp"
#include "HostDeviceArray.hpp"
#include "Operator.hpp"
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

        /// @brief Returns the diagonal mass vector (length fem.size()) on the device.
        const_dvec_wrapper to_device() const { return reshape(_m.device_read(), _m.size()); }

    private:
        template <typename Func>
        friend void h1_trace(const FaceMassMatrix3D &, const Func &, double *);

        const H1Space3D &_fem;
        HostDeviceArray<double> _m;
    };

    /**
     * @brief Computes F[i] = <f, phi[i]> for all i in H1Space3D.
     * Interior DOF entries of F are zero; boundary DOF entries are
     * f(x_i) * m_i where m_i is the face mass weight at node i.
     */
    template <typename Func>
    void h1_trace(const FaceMassMatrix3D &H, const Func &f, double *F)
    {
        const int ndof = H._fem.size();
        auto m = H._m.device_read();
        auto x = H._fem.physical_coordinates(MemorySpace::DEVICE);

        forall(ndof, [=] __device__(int i) -> void { F[i] = m[i] * f(x[i]); });
    }

    /**
     * @brief Computes F[i] = f(r[i]) for all i, where r[i] is the i-th physical coordinate in the TraceSpace3D.
     * @param tr TraceSpace3D
     * @param f a function f(double3) -> double
     * @param F a vector in the TraceSpace3D. On exit, F[i] = f(r[i]).
     */
    template <typename Func>
    void trace(const TraceSpace3D &tr, const Func &f, double *F)
    {
        const int ndof = tr.size();
        auto x = tr.h1_space().physical_coordinates(MemorySpace::DEVICE);
        auto global_indices = tr.global_indices(MemorySpace::DEVICE);

        forall(ndof, [=] __device__(int i) -> void {
            const double3 r = x[global_indices[i]];
            F[i] = f(r);
        });
    }
} // namespace cuddh
