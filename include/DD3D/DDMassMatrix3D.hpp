#ifndef DDH_DD_MASS_MATRIX_3D_HPP
#define DDH_DD_MASS_MATRIX_3D_HPP

#include "cuddh_config.hpp"
#include "cuddh_error.hpp"
#include "EnsembleSpace3D.hpp"

#include <thrust/universal_vector.h>
#include "forall.hpp"

namespace cuddh
{
    class DDMassMatrix3D
    {
    public:
        DDMassMatrix3D(const H1Space3D &fem, const EnsembleSpace3D &efem);

        auto to_device() const
        {
            return reshape(m, mx_dofs, n_domains);
        }

    private:
        int mx_dofs;
        int n_domains;
        thrust::universal_vector<float> m;
    };
} // namespace cuddh

#endif
