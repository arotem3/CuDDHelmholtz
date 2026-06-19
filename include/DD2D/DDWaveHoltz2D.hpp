#pragma once

#include <thrust/extrema.h>

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
    DDWaveHoltz<T> make_DDWaveHoltz_2d(const EnsembleSpace &efem, T omega, const GridFunc2D<double> *a = nullptr);

    extern template DDWaveHoltz<float> make_DDWaveHoltz_2d<float>(const EnsembleSpace &efem, float omega,
                                                                  const GridFunc2D<double> *a);
    extern template DDWaveHoltz<double> make_DDWaveHoltz_2d<double>(const EnsembleSpace &efem, double omega,
                                                                    const GridFunc2D<double> *a);
} // namespace cuddh
