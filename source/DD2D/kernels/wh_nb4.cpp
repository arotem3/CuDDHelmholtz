#include "DDWHDispatcherImpl.hpp"

namespace cuddh::details
{
    DECLARE_WH_KERNELS_FOR_TDOF(, float, 4);
    DECLARE_WH_KERNELS_FOR_TDOF(, double, 4);
} // namespace cuddh::details
