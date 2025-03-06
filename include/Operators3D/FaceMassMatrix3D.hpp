#ifndef CUDDH_FACE_MASS_MATRIX_HPP
#define CUDDH_FACE_MASS_MATRIX_HPP

#include "H1Space3D.hpp"
#include "Operator.hpp"
#include "HostDeviceArray.hpp"
#include "forall.hpp"
#include "linalg.hpp"

namespace cuddh
{
    /**
     * @brief m(u, phi) = (a(x) * u, phi) for all phi in a FaceSpace3D
     */
    class FaceMassMatrix3D : public Operator
    {
    public:
        FaceMassMatrix3D(const TraceSpace3D& tr);
        FaceMassMatrix3D(const double * a, const TraceSpace3D& tr);

        /**
         * @brief y[i] <- y[i] + c * (x, phi[i]),
         * where phi[i] is the i-th basis function in the TraceSpace3D.
         * @param c scalar coefficient
         * @param x a vector in the TraceSpace3D
         * @param y a vector in the TraceSpace3D. On exit, y[i] <- y[i] + c * (x, phi[i]).
         */
        void action(double c, const double * x, double * y) const override;

        /**
         * @brief y[i] = (x, phi[i])
         * @param x a vector in the TraceSpace3D
         * @param y a vector in the TraceSpace3D. On exit, y[i] = (x, phi[i]).
         */
        void action(const double * x, double * y) const override;
    
    private:
        const TraceSpace3D& tr;
        host_device_dvec m;
    };
} // namespace cuddh

#endif
