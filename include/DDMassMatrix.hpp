#ifndef DDH_DD_MASS_MATRIX_HPP
#define DDH_DD_MASS_MATRIX_HPP

#include "cuddh_config.hpp"
#include "cuddh_error.hpp"
#include "EnsembleSpace.hpp"

#include "HostDeviceArray.hpp"
#include "forall.hpp"

namespace cuddh
{
    class DDMassMatrix
    {
    public:
        DDMassMatrix(const H1Space &fem, const EnsembleSpace &efem);

        auto to_device() const
        {
            return reshape(m.device_read(), mx_dofs, n_domains);
        }

    private:
        int mx_dofs;
        int n_domains;
        HostDeviceArray<float> m;
    };
} // namespace cuddh

#endif