#pragma once

#include <thrust/extrema.h>

#include "DDWaveHoltz.hpp"
#include "EnsembleSpace3D.hpp"
#include "FEM3D/GridFunc3D.hpp"

namespace cuddh
{
    template <typename T>
    DDWaveHoltz<T> make_DDWaveHoltz_3d(const EnsembleSpace3D &efem, T omega, const GridFunc3D<double> *a);
    extern template DDWaveHoltz<float> make_DDWaveHoltz_3d<float>(const EnsembleSpace3D &efem, float omega,
                                                                  const GridFunc3D<double> *a);
    extern template DDWaveHoltz<double> make_DDWaveHoltz_3d<double>(const EnsembleSpace3D &efem, double omega,
                                                                    const GridFunc3D<double> *a);
} // namespace cuddh
