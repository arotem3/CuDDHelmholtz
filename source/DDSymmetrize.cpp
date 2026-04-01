#include "DDSymmetrize.hpp"

using namespace cuddh;

template <typename scalar_t>
static void _symmetrize_ddh(int n, const scalar_t *x, scalar_t *y)
{
    constexpr scalar_t half(0.5);
    const int m = n / 2;

    forall(m, [=] __device__(const int i) mutable -> void {
        const int inds[] = {i, i + m, n + i, n + m + i};

        scalar_t Y[4];
        for (int j = 0; j < 4; ++j)
        {
            Y[j] = y[inds[j]];
            if (x)
                Y[j] = x[inds[j]] - Y[j];
        }

        scalar_t RY[] = {half * (Y[0] + Y[1] + Y[2] - Y[3]), half * (Y[0] + Y[1] - Y[2] + Y[3]),
                         -half * (-Y[0] + Y[1] + Y[2] + Y[3]), -half * (Y[0] - Y[1] + Y[2] + Y[3])};

        for (int j = 0; j < 4; ++j)
            y[inds[j]] = RY[j];
    });
}

void cuddh::symmetrize_ddh(int n, const float *x, float *y)
{
    _symmetrize_ddh<float>(n, x, y);
}
void cuddh::symmetrize_ddh(int n, const double *x, double *y)
{
    _symmetrize_ddh<double>(n, x, y);
}
