#include "DDMRDispatcherImpl.hpp"

namespace cuddh::details
{
    DECLARE_MR_KERNELS_FOR_TDOF(, float, 8);
    DECLARE_MR_KERNELS_FOR_TDOF(, double, 8);
} // namespace cuddh::details
