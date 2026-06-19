#include "DDMRDispatcherImpl.hpp"

namespace cuddh::details
{
    DECLARE_MR_KERNELS_FOR_TDOF(, float, 2);
    DECLARE_MR_KERNELS_FOR_TDOF(, double, 2);
} // namespace cuddh::details
