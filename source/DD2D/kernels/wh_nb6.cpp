#include "DDWHDispatcherImpl.hpp"

namespace cuddh::details
{
    DECLARE_WH_KERNELS_FOR_TDOF(, float, 6);
    DECLARE_WH_KERNELS_FOR_TDOF(, double, 6);
} // namespace cuddh::details
