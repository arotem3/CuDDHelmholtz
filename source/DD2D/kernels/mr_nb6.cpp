#include "DDMRDispatcherImpl.hpp"

namespace cuddh::details
{
    DECLARE_MR_KERNELS_FOR_TDOF(, float, 6);
    DECLARE_MR_KERNELS_FOR_TDOF(, double, 6);
} // namespace cuddh::details
