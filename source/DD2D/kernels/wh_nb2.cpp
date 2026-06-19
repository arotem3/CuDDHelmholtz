#include "DDWHDispatcherImpl.hpp"

namespace cuddh::details
{
    DECLARE_WH_KERNELS_FOR_TDOF(, float, 2);
    DECLARE_WH_KERNELS_FOR_TDOF(, double, 2);
} // namespace cuddh::details