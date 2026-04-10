#include "DDMRDispatcherImpl.hpp"

namespace cuddh::details
{
    DECLARE_MR_KERNELS_FOR_TDOF(, float, 3);
    DECLARE_MR_KERNELS_FOR_TDOF(, double, 3);
} // namespace cuddh::details
