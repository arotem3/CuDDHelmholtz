#include "cuddh_error.hpp"

namespace cuddh
{
    __host__ __device__ void cuddh_error(const char * msg)
    {
        printf("--- CUDDH ERROR ---\n\t%s\n-------------------\n", msg);
        assert(0);
    }
} // namespace cuddh
