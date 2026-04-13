#pragma once

#include <thrust/universal_vector.h>

#include "EnsembleSpace3D.hpp"
#include "cuddh_config.hpp"
#include "cuddh_error.hpp"

namespace cuddh
{
    class DDFaceMassMatrix3D
    {
    public:
        DDFaceMassMatrix3D(const H1Space3D &tr, const EnsembleSpace3D &efem);

        auto to_device() const { return reshape(m, mx_fdofs, n_domains); }

    private:
        int mx_fdofs;
        int n_domains;
        thrust::universal_vector<float> m;
    };
} // namespace cuddh
