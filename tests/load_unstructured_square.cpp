#include "test.hpp"

using namespace cuddh;

namespace cuddh_test
{
    Mesh2D load_unstructured_square()
    {
        std::string dir = "meshes/unstructured_square";
        std::ifstream info(dir + "/info.txt");
        if (not info)
        {
            std::string err = "cuddh_test::load_unstructured_square() error: cannot open file: " + dir + "/info.txt";
            cuddh_error(err.c_str());
        }

        int n_pts, n_elem;
        info >> n_pts >> n_elem;
        info.close();

        dmat x(2, n_pts);
        imat elems(4, n_elem);

        std::ifstream coo(dir + "/coordinates.txt");
        if (not coo)
        {
            std::string err = "cuddh_test::load_unstructured_square() error: cannot open file: " + dir + "/coordinates.txt";
            cuddh_error(err.c_str());
        }

        for (int i = 0; i < n_pts; ++i)
        {
            coo >> x(0, i) >> x(1, i);
        }
        coo.close();

        std::ifstream elements(dir + "/elements.txt");
        if (not coo)
        {
            std::string err = "cuddh_test::load_unstructured_square() error: cannot open file: " + dir + "/elements.txt";
            cuddh_error(err.c_str());
        }

        for (int i = 0; i < n_elem; ++i)
        {
            elements >> elems(0, i) >> elems(1, i) >> elems(2, i) >> elems(3, i);
        }
        elements.close();

        return Mesh2D::from_vertices(n_pts, x.data(), n_elem, elems.data());
    }
} // namespace cuddh_test
