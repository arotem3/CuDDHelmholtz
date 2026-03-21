#ifndef DDH_DD_STIFFNESS_MATRIX_3D_HPP
#define DDH_DD_STIFFNESS_MATRIX_3D_HPP

#include "cuddh_config.hpp"
#include "cuddh_error.hpp"
#include "EnsembleSpace3D.hpp"
#include "SmallMatrix.hpp"

#include <thrust/universal_vector.h>
#include "forall.hpp"

namespace cuddh
{
    class DDStiffnessMatrix3D
    {
    public:
        struct DeviceDDStiffnessMatrix3D
        {
            MatrixWrapper<const float> D;
            TensorWrapper<5, const fsym3x3> G;
        };

        DDStiffnessMatrix3D(const H1Space3D &fem, const EnsembleSpace3D &efem);

        DeviceDDStiffnessMatrix3D to_device() const
        {
            auto D = reshape(d, n_basis, n_basis);
            auto G = reshape(g, n_basis, n_basis, n_basis, mx_elem, n_domains);
            return {D, G};
        }

    private:
        int n_basis;
        int mx_elem;
        int n_domains;
        thrust::universal_vector<float> d;
        thrust::universal_vector<fsym3x3> g;
    };
} // namespace cuddh


#endif
