#ifndef CUDDH_DD_FACE_MASS_3D_HPP
#define CUDDH_DD_FACE_MASS_3D_HPP

#include "cuddh_config.hpp"
#include "cuddh_error.hpp"
#include "EnsembleSpace3D.hpp"

#include <thrust/universal_vector.h>
#include "forall.hpp"

namespace cuddh
{
    class DDFaceMassMatrix3D
    {
    public:
        DDFaceMassMatrix3D(const H1Space3D &tr, const EnsembleSpace3D &efem);

        auto to_device() const
        {
            return reshape(m, mx_fdofs, n_domains);
        }
    
    private:
        int mx_fdofs;
        int n_domains;
        thrust::universal_vector<float> m;
    };
} // namespace cuddh


#endif
