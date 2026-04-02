#pragma once

#include <thrust/extrema.h>

#include "DDFaceMassMatrix3D.hpp"
#include "DDMassMatrix3D.hpp"
#include "DDStiffnessMatrix3D.hpp"
#include "DDWaveHoltz.hpp"
#include "EnsembleSpace3D.hpp"
#include "HostDeviceArray.hpp"
#include "Operators3D/MassMatrix3D.hpp"
#include "cuddh_config.hpp"
#include "cuddh_error.hpp"
#include "cxmult.hpp"
#include "forall.hpp"
#include "linalg.hpp"

namespace cuddh
{
    template <typename T>
    DDWaveHoltz<T> make_DDWaveHoltz_3d(T omega, const double *a, const H1Space3D &fem, const EnsembleSpace3D &efem);
    extern template DDWaveHoltz<float> make_DDWaveHoltz_3d<float>(float omega, const double *a, const H1Space3D &fem,
                                                                  const EnsembleSpace3D &efem);
    extern template DDWaveHoltz<double> make_DDWaveHoltz_3d<double>(double omega, const double *a, const H1Space3D &fem,
                                                                    const EnsembleSpace3D &efem);
} // namespace cuddh
