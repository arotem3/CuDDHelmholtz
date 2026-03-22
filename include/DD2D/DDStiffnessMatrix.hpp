#pragma once

#include <type_traits>

#include "EnsembleSpace.hpp"
#include "HostDeviceArray.hpp"
#include "SmallMatrix.hpp"
#include "cuddh_config.hpp"
#include "cuddh_error.hpp"
#include "forall.hpp"

namespace cuddh
{
    template <typename scalar_t>
    class DDStiffnessMatrix
    {
        static_assert(std::is_same_v<scalar_t, float> || std::is_same_v<scalar_t, double>,
                      "scalar_t must be float or double");

    public:
        using sym2x2 = SmallSymmetricMatrix<scalar_t, 2>;

        struct DeviceDDStiffnessMatrix
        {
            MatrixWrapper<const scalar_t> D;
            TensorWrapper<4, const sym2x2> G;
        };

        DDStiffnessMatrix(const H1Space2D &fem, const EnsembleSpace &efem);

        DeviceDDStiffnessMatrix to_device() const
        {
            auto D = reshape(d.device_read(), n_basis, n_basis);
            auto G = reshape(g.device_read(), n_basis, n_basis, mx_elem, n_domains);
            return {D, G};
        }

    private:
        int n_basis;
        int mx_elem;
        int n_domains;
        HostDeviceArray<scalar_t> d;
        HostDeviceArray<sym2x2> g;
    };

    extern template class DDStiffnessMatrix<float>;
    extern template class DDStiffnessMatrix<double>;
} // namespace cuddh
