#include "DDMRDispatcherImpl.hpp"

namespace cuddh::details
{
    DECLARE_MR_KERNELS_FOR_TDOF(, float, 4);
    DECLARE_MR_KERNELS_FOR_TDOF(, double, 4);
} // namespace cuddh::details
