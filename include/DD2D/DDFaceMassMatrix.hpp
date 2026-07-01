#pragma once

#include <complex>

#include "EnsembleSpace.hpp"
#include "FEM2D/GridFunc2D.hpp"
#include "HostDeviceArray.hpp"
#include "Tensor.hpp"
#include "cuddh_config.hpp"
#include "cuddh_error.hpp"

namespace cuddh
{
    template <typename scalar_t, bool Complex>
    class BlockSparseMatrix;

    template <typename scalar_t>
    class DDFaceMassMatrix
    {
        static_assert(std::is_same_v<scalar_t, float> || std::is_same_v<scalar_t, double>,
                      "scalar_t must be float or double");

    public:
        DDFaceMassMatrix(const EnsembleSpace &efem);
        DDFaceMassMatrix(const EnsembleSpace &efem, const GridFunc2D<double> &a);

        auto to_device() const { return reshape(m.device_read(), mx_fdof, n_domains); }

        /// @brief Accumulate c * H_p into the diagonal of block p of B for each subdomain p,
        /// for the face DOFs (indices 0..fsizes(p)-1 within the volume DOF space).
        /// B must be in COOAssembly state (after `finalize_pattern()`).
        void assemble(scalar_t c, BlockSparseMatrix<scalar_t, false> &B) const;
        void assemble(std::complex<scalar_t> c, BlockSparseMatrix<scalar_t, true> &B) const;

    private:
        const EnsembleSpace &efem;
        int mx_fdof;
        int n_domains;
        HostDeviceArray<scalar_t> m;
    };

    extern template class DDFaceMassMatrix<float>;
    extern template class DDFaceMassMatrix<double>;
} // namespace cuddh
