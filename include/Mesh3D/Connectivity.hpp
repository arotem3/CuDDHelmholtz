#ifndef CUDDH_MESH3D_CONNECTIVITY_HPP
#define CUDDH_MESH3D_CONNECTIVITY_HPP

#include "cuddh_config.hpp"

namespace cuddh
{
    struct FaceConnectivity
    {
        enum Permutation
        {
            Identity = 0,
            Rotate90 = 1,
            Rotate180 = 2,
            Rotate270 = 3,
            HorizontalFlip = 4,
            VerticalFlip = 5,
            DiagonalFlipMain = 6,
            DiagonalFlipAnti = 7
        };

        enum Label
        {
            ZMin = 0,
            ZMax = 1,
            XMin = 2,
            XMax = 3,
            YMin = 4,
            YMax = 5,
            None = -1
        };

        int elements[2];
        Label label[2];
        Permutation permutation;
    };
} // namespace cuddh

#endif
