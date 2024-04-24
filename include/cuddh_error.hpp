#ifndef CUDDH_ERROR_HPP
#define CUDDH_ERROR_HPP

#include <string>
#include <stdexcept>
#include <iostream>
#include <assert.h>

#include "cuddh_config.hpp"

namespace cuddh
{
    __host__ __device__ void cuddh_error(const char *);
} // namespace cuddh

#endif
