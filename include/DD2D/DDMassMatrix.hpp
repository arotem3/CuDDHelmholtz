#pragma once

#include "EnsembleSpace.hpp"
#include "FEM2D/GridFunc2D.hpp"
#include "HostDeviceArray.hpp"
#include "cuddh_config.hpp"
#include "cuddh_error.hpp"
#include "forall.hpp"

namespace cuddh
{
    template <typename scalar_t>
    class DDMassMatrix
    {
        static_assert(std::is_same_v<scalar_t, float> || std::is_same_v<scalar_t, double>,
                      "scalar_t must be float or double");

    public:
        DDMassMatrix(const EnsembleSpace &efem);
        DDMassMatrix(const EnsembleSpace &efem, const GridFunc2D<double> &a);

        auto to_device() const { return reshape(m.device_read(), mx_dofs, n_domains); }

    private:
        int mx_dofs;
        int n_domains;
        HostDeviceArray<scalar_t> m;
    };

    extern template class DDMassMatrix<float>;
    extern template class DDMassMatrix<double>;
} // namespace cuddh
