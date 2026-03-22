#pragma once

namespace cuddh::details
{
    // computes the complex multiplication (c + i*s) * (x + i*y) and stores the result in x and y.
    template <typename scalar_t>
    inline constexpr void cxmult(scalar_t &x, scalar_t &y, scalar_t c, scalar_t s)
    {
        scalar_t t = x;
        x = c * t - s * y;
        y = s * t + c * y;
    }
} // namespace cuddh::details
