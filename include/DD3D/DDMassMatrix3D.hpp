#pragma once

#include <thrust/universal_vector.h>

#include "EnsembleSpace3D.hpp"
#include "FEM3D/GridFunc3D.hpp"
#include "Tensor.hpp"
#include "cuddh_config.hpp"
#include "cuddh_error.hpp"

namespace cuddh
{
    template <typename scalar_t>
    class DDMassMatrix3D
    {
        static_assert(std::is_same_v<scalar_t, float> || std::is_same_v<scalar_t, double>,
                      "scalar_t must be float or double");

    public:
        DDMassMatrix3D(const EnsembleSpace3D &efem);
        DDMassMatrix3D(const EnsembleSpace3D &efem, const GridFunc3D<double> &a);

        MatrixWrapper<const scalar_t> to_device() const { return reshape(m, mx_dofs, n_domains); }

    private:
        int mx_dofs;
        int n_domains;
        thrust::universal_vector<scalar_t> m;
    };

    extern template class DDMassMatrix3D<float>;
    extern template class DDMassMatrix3D<double>;
} // namespace cuddh
