#pragma once

#include "DDFaceMassMatrix.hpp"
#include "DDMassMatrix.hpp"
#include "DDStiffnessMatrix.hpp"
#include "DDWaveHoltz.hpp"
#include "EnsembleSpace.hpp"
#include "HostDeviceArray.hpp"
#include "Operator.hpp"
#include "cuddh_config.hpp"
#include "cuddh_error.hpp"
#include "cxmult.hpp"
#include "forall.hpp"
#include "linalg.hpp"

namespace cuddh
{
    template <typename T>
    DDWaveHoltz<T> make_DDWaveHoltz_2d(T omega, const double *a, const H1Space2D &fem, const EnsembleSpace &efem);
    extern template DDWaveHoltz<float> make_DDWaveHoltz_2d<float>(float omega, const double *a, const H1Space2D &fem,
                                                                  const EnsembleSpace &efem);
    extern template DDWaveHoltz<double> make_DDWaveHoltz_2d<double>(double omega, const double *a, const H1Space2D &fem,
                                                                    const EnsembleSpace &efem);
} // namespace cuddh
