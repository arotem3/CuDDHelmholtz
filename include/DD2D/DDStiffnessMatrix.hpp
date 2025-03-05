#ifndef CUDDH_DD_STIFFNESS_MATRIX_HPP
#define CUDDH_DD_STIFFNESS_MATRIX_HPP

#include "cuddh_config.hpp"
#include "cuddh_error.hpp"
#include "EnsembleSpace.hpp"

#include "HostDeviceArray.hpp"
#include "forall.hpp"

namespace cuddh
{
    class DDStiffnessMatrix
    {
    public:
        struct DeviceDDStiffnessMatrix
        {
            MatrixWrapper<const float> D;
            MatrixWrapper<const float3> G;
        };

        DDStiffnessMatrix(const H1Space &fem, const EnsembleSpace &efem);

        DeviceDDStiffnessMatrix to_device() const
        {
            auto D = reshape(d.device_read(), n_basis, n_basis);
            auto G = reshape(g.device_read(), n_basis * n_basis * mx_elem, n_domains);
            return {D, G};
        }

    private:
        int n_basis;
        int mx_elem;
        int n_domains;
        HostDeviceArray<float> d;
        HostDeviceArray<float3> g;
    };
} // namespace cuddh

#endif