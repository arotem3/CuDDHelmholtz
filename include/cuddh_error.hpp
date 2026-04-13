#pragma once

#include <assert.h>

#include <iostream>
#include <stdexcept>
#include <string>

#include "cuddh_config.hpp"

/* Macros for error checking and assertions
 * cuddh_verify: checks a condition and reports an error message if the condition is false. Runs in release.
 * cuddh_assert: checks a condition and reports an error message if the condition is false. Only active in debug builds.
 */

#ifdef __CUDA_ARCH__
#define cuddh_verify(cond, msg)                                           \
    do                                                                    \
    {                                                                     \
        if (!(cond))                                                      \
        {                                                                 \
            printf("CUDDH device error at %s:%d:\n", __FILE__, __LINE__); \
            msg;                                                          \
            asm("trap;");                                                 \
        }                                                                 \
    } while (0)
#else
#define cuddh_verify(cond, msg)                                         \
    do                                                                  \
    {                                                                   \
        if (!(cond))                                                    \
        {                                                               \
            printf("CUDDH host error at %s:%d:\n", __FILE__, __LINE__); \
            msg;                                                        \
            std::abort();                                               \
        }                                                               \
    } while (0)
#endif

#ifdef CUDDH_DEBUG
#define cuddh_assert(cond, msg) cuddh_verify(cond, msg)
#else
#define cuddh_assert(cond, msg)
#endif
