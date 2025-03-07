#ifndef CUDDH_MASS_MATRIX_3D_HPP
#define CUDDH_MASS_MATRIX_3D_HPP

#include "H1Space3D.hpp"
#include "Operator.hpp"

#include "HostDeviceArray.hpp"
#include "forall.hpp"
#include "linalg.hpp"

namespace cuddh
{
    class InvMassMatrix3D;

    /**
     * @brief spectral element mass matrix (a(x) * u, v) or (u, v) in 3D.
     */
    class MassMatrix3D : public Operator
    {
    public:
        /**
         * @brief initialize mass matrix m(u, v) = (u, v)
         */
        MassMatrix3D(const H1Space3D &fem);

        /**
         * @brief initialize weighted mass matrix m(u, v) = (a(x)*u, v)
         * @param a DEVICE. H1Space3D vector representing the function a.
         * @param fem
         */
        MassMatrix3D(const double *d_a, const H1Space3D &fem);

        /**
         * @brief y <- y + c * M*x, where M is the mass matrix
         */
        void action(double c, const double *x, double *y) const override;

        /**
         * @brief y <- M*x, where M is the mass matrix
         */
        void action(const double *x, double *y) const override;

        InvMassMatrix3D inv() const;

    private:
        friend class InvMassMatrix3D;
        
        template <typename Func>
        friend void l2_project(const MassMatrix3D &M, const Func &f, double *F);

        friend double l2_dot(const MassMatrix3D &M, const double *x, const double *y);
        friend double l2_norm(const MassMatrix3D &M, const double *x);
        friend double l2_dist(const MassMatrix3D &M, const double *x, const double *y);

        const H1Space3D &fem;
        host_device_dvec _m; // a(x) * w(i) * w(j) * w(k) * detJ
    };

    /**
     * @brief inverse of mass matrix in 3D
     */
    class InvMassMatrix3D : public Operator
    {
    public:
        InvMassMatrix3D(const H1Space3D &fem);
        InvMassMatrix3D(const double *d_a, const H1Space3D &fem);
        InvMassMatrix3D(const MassMatrix3D &M);

        /**
         * @brief y <- y + c * inv(M)*x, where M is the mass matrix.
         */
        void action(double c, const double *x, double *y) const override;

        /**
         * @brief y <- inv(M)*x, where M is the mass matrix.
         */
        void action(const double *x, double *y) const override;

    private:
        const H1Space3D &fem;
        host_device_dvec _mi; // 1 / ( a(x) * w(i) * w(j) * w(k) * detJ )
    };

    /**
     * @brief computes F[i] = (f, phi_i) for i = 0, 1, ..., N-1
     * @param M mass matrix
     * @param f DEVICE. function f(double3 x) to project
     * @param F DEVICE. output array of size N
     */
    template <typename Func>
    void l2_project(const MassMatrix3D &M, const Func &f, double *F)
    {
        const int ndof = M.fem.size();

        auto x = M.fem.physical_coordinates(MemorySpace::DEVICE);
        auto m = M._m.read(MemorySpace::DEVICE);

        forall(ndof, [=] __device__(int i)
        {
            double3 xi = x[i];
            F[i] = f(xi) * m[i];
        });
    }

    double l2_dot(const MassMatrix3D &M, const double *x, const double *y);
    inline double l2_norm(const MassMatrix3D &M, const double *x) { return std::sqrt(l2_dot(M, x, x)); }
    double l2_dist(const MassMatrix3D &M, const double *x, const double *y);
} // namespace cuddh

#endif
