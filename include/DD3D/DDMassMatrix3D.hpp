#pragma once

#include <thrust/universal_vector.h>

#include "EnsembleSpace3D.hpp"
#include "FEM3D/GridFunc3D.hpp"
#include "Tensor.hpp"
#include "cuddh_config.hpp"
#include "cuddh_error.hpp"

namespace cuddh
{
    class DDMassMatrix3D
    {
    public:
        DDMassMatrix3D(const EnsembleSpace3D &efem);
        DDMassMatrix3D(const EnsembleSpace3D &efem, const GridFunc3D<double> &a);

        MatrixWrapper<const float> to_device() const { return reshape(m, mx_dofs, n_domains); }

    private:
        int mx_dofs;
        int n_domains;
        thrust::universal_vector<float> m;
    };
} // namespace cuddh
