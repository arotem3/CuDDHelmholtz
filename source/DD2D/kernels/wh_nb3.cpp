#include "DDWHDispatcherImpl.hpp"

namespace cuddh::details
{
    DECLARE_WH_KERNELS_FOR_TDOF(, float, 3);
    DECLARE_WH_KERNELS_FOR_TDOF(, double, 3);
} // namespace cuddh::details
