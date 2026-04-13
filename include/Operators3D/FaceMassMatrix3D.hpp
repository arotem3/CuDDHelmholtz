#pragma once

#include "FEM3D/H1Space3D.hpp"
#include "HostDeviceArray.hpp"
#include "Operator.hpp"
#include "forall.hpp"

namespace cuddh
{
    /**
     * @brief m(u, phi) = (a(x) * u, phi) for all phi in a FaceSpace3D
     */
    class FaceMassMatrix3D : public Operator<double>
    {
    public:
        FaceMassMatrix3D(const TraceSpace3D &tr);
        FaceMassMatrix3D(const double *a, const TraceSpace3D &tr);

        /**
         * @brief y[i] <- y[i] + c * (x, phi[i]),
         * where phi[i] is the i-th basis function in the TraceSpace3D.
         * @param c scalar coefficient
         * @param x a vector in the TraceSpace3D
         * @param y a vector in the TraceSpace3D. On exit, y[i] <- y[i] + c * (x, phi[i]).
         */
        void action(double c, const double *x, double *y) const override;

        /**
         * @brief y[i] = (x, phi[i])
         * @param x a vector in the TraceSpace3D
         * @param y a vector in the TraceSpace3D. On exit, y[i] = (x, phi[i]).
         */
        void action(const double *x, double *y) const override;

    private:
        template <typename Func>
        friend void h1_trace(const FaceMassMatrix3D &, const Func &, double *);

        const TraceSpace3D &tr;
        host_device_dvec m;
    };

    /**
     * @brief Computes F[i] = (f, phi[i]) for all i, where phi[i] is the i-th basis function in the TraceSpace3D.
     * @param H FaceMassMatrix3D
     * @param f a function f(double3) -> double
     * @param F a vector in the TraceSpace3D. On exit, F[i] = (f, phi[i]).
     */
    template <typename Func>
    void h1_trace(const FaceMassMatrix3D &H, const Func &f, double *F)
    {
        const int ndof = H.tr.size();

        auto m = H.m.read(MemorySpace::DEVICE);
        auto x = H.tr.h1_space().physical_coordinates(MemorySpace::DEVICE);
        auto global_indices = H.tr.global_indices(MemorySpace::DEVICE);

        forall(ndof, [=] __device__(int i) -> void {
            const double3 r = x[global_indices[i]];
            F[i] = m[i] * f(r);
        });
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
