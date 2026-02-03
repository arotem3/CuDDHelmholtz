#include "test.hpp"

#ifndef UNSTRUCTURED_SQUARE_MESH_DIR
#error "UNSTRUCTURED_SQUARE_MESH_DIR is not defined: this should be defined by the build system"
#endif

using namespace cuddh;

namespace cuddh_test
{
    Mesh2D load_unstructured_square()
    {
        std::string dir = UNSTRUCTURED_SQUARE_MESH_DIR;
        std::ifstream info(dir + "/info.txt");
        cuddh_verify(info, printf("cuddh_test::load_unstructured_square() error: cannot open file: %s/info.txt", dir.c_str()));

        int n_pts, n_elem;
        info >> n_pts >> n_elem;
        info.close();

        dmat x(2, n_pts);
        imat elems(4, n_elem);

        std::ifstream coo(dir + "/coordinates.txt");
        cuddh_verify(coo, printf("cuddh_test::load_unstructured_square() error: cannot open file: %s/coordinates.txt", dir.c_str()));

        for (int i = 0; i < n_pts; ++i)
        {
            coo >> x(0, i) >> x(1, i);
        }
        coo.close();

        std::ifstream elements(dir + "/elements.txt");
        cuddh_verify(elements, printf("cuddh_test::load_unstructured_square() error: cannot open file: %s/elements.txt", dir.c_str()));

        for (int i = 0; i < n_elem; ++i)
        {
            elements >> elems(0, i) >> elems(1, i) >> elems(2, i) >> elems(3, i);
        }
        elements.close();

        return Mesh2D::from_vertices(n_pts, x.data(), n_elem, elems.data());
    }
} // namespace cuddh_test
