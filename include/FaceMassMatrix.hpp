#ifndef CUDDH_FACE_MASS_MATRIX_HPP
#define CUDDH_FACE_MASS_MATRIX_HPP

#include "H1Space.hpp"
#include "Operator.hpp"
#include "HostDeviceArray.hpp"
#include "forall.hpp"

namespace cuddh
{
    /// @brief m(u, phi) = (a(x) * u, phi) for all phi in a FaceSpace
    class FaceMassMatrix : public Operator
    {
    public:
        FaceMassMatrix(const FaceSpace& fs);
        FaceMassMatrix(const double * a, const FaceSpace& fs);

        /// @brief y[i] <- y[i] + c * (x, phi[i]),
        /// where phi[i] is the i-th basis function in the FaceSpace.
        /// @param c scalar coefficient
        /// @param x a vector in the FaceSpace
        /// @param y a vector in the FaceSpace. On exit, y[i] <- y[i] + c * (x, phi[i]).
        void action(double c, const double * x, double * y) const override;

        /// @brief y[i] = (x, phi[i])
        void action(const double * x, double * y) const override;

    private:
        const FaceSpace& fs;

        const int ndof;
        const int n_faces;
        const int n_basis;
        const int n_quad;

        host_device_dvec _a;
        host_device_dvec _P;
    };

    /// @brief diagonal approximate inverse of FaceMassMatrix
    class DiagInvFaceMassMatrix : public Operator
    {
    public:
        DiagInvFaceMassMatrix(const FaceSpace& fs);
        DiagInvFaceMassMatrix(const double * a, const FaceSpace& fs);

        /// @brief y <- y + c * A * x where A ~ inv(M).
        /// @param c scalar coefficient
        /// @param x FaceSpace vector
        /// @param y FaceSpace vector
        void action(double c, const double * x, double * y) const override;

        void action(const double * x, double * y) const override;

    private:
        const int ndof;
        host_device_dvec inv_m;
    };
} // namespace cuddh

#endif
