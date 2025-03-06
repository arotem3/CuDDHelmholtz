#ifndef CUDDH_DD_FACE_MASS_MATRIX_HPP
#define CUDDH_DD_FACE_MASS_MATRIX_HPP

#include "cuddh_config.hpp"
#include "cuddh_error.hpp"
#include "EnsembleSpace.hpp"

#include "HostDeviceArray.hpp"
#include "forall.hpp"

namespace cuddh
{
    class DDFaceMassMatrix
    {
    public:
        DDFaceMassMatrix(const H1Space2D &fem, const EnsembleSpace &efem);

        auto to_device() const
        {
            return reshape(m.device_read(), mx_fdof, n_domains);
        }

    private:
        int mx_fdof;
        int n_domains;
        HostDeviceArray<float> m;
    };
} // namespace cuddh

#endif