#pragma once

#include "EnsembleSpace.hpp"
#include "HostDeviceArray.hpp"
#include "cuddh_config.hpp"
#include "cuddh_error.hpp"
#include "forall.hpp"

namespace cuddh
{
    template <typename scalar_t>
    class DDFaceMassMatrix
    {
        static_assert(std::is_same_v<scalar_t, float> || std::is_same_v<scalar_t, double>,
                      "scalar_t must be float or double");

    public:
        DDFaceMassMatrix(const H1Space2D &fem, const EnsembleSpace &efem);

        auto to_device() const { return reshape(m.device_read(), mx_fdof, n_domains); }

    private:
        int mx_fdof;
        int n_domains;
        HostDeviceArray<scalar_t> m;
    };

    extern template class DDFaceMassMatrix<float>;
    extern template class DDFaceMassMatrix<double>;
} // namespace cuddh
