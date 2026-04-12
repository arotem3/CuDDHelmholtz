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

    /**
     * @brief returns the permutation of the indices (i, j) of a face degree of freedom.
     *
     * @param N the number of basis functions in each direction
     * @param i the row index of the face degree of freedom
     * @param j the column index of the face degree of freedom
     * @param p the permutation of the face degree of freedom
     * @return std::pair<int, int> the permuted indices
     */
    inline constexpr std::array<int, 2> permute_face_index(int N, int i, int j, FaceConnectivity::Permutation p)
    {
        switch (p)
        {
            case FaceConnectivity::Permutation::Rotate90:
                return {j, N - 1 - i};
            case FaceConnectivity::Permutation::Rotate180:
                return {N - 1 - i, N - 1 - j};
            case FaceConnectivity::Permutation::Rotate270:
                return {N - 1 - j, i};
            case FaceConnectivity::Permutation::HorizontalFlip:
                return {N - 1 - i, j};
            case FaceConnectivity::Permutation::VerticalFlip:
                return {i, N - 1 - j};
            case FaceConnectivity::Permutation::DiagonalFlipMain:
                return {j, i};
            case FaceConnectivity::Permutation::DiagonalFlipAnti:
                return {N - 1 - j, N - 1 - i};
            default: // Identity
                return {i, j};
        }
    }

    /**
     * @brief returns the global index of a face degree of freedom.
     *
     */
    inline constexpr std::array<int, 3> face2vol(int N, int i, int j, FaceConnectivity::Label f)
    {
        int m = 0, n = 0, l = 0;

        if (f == FaceConnectivity::Label::ZMin || f == FaceConnectivity::Label::ZMax)
        {
            m = i;
            n = j;
            l = (f == FaceConnectivity::Label::ZMin) ? 0 : (N - 1);
        }
        else if (f == FaceConnectivity::Label::XMin || f == FaceConnectivity::Label::XMax)
        {
            m = (f == FaceConnectivity::Label::XMin) ? 0 : (N - 1);
            n = i;
            l = j;
        }
        else // YMin or YMax
        {
            m = i;
            n = (f == FaceConnectivity::Label::YMin) ? 0 : (N - 1);
            l = j;
        }

        return {m, n, l};
    }
} // namespace cuddh

#endif
