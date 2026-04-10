#include "DDWHDispatcherImpl.hpp"

namespace cuddh::details
{
    DECLARE_WH_KERNELS_FOR_TDOF(, float, 8);
    DECLARE_WH_KERNELS_FOR_TDOF(, double, 8);
} // namespace cuddh::details
