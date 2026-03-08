#pragma once

#include <format>
#include <fstream>
#include <iostream>
#include <string>
#include <string_view>

#include "cuddh.hpp"

struct TestLogger
{
    int total = 0;
    int passed = 0;

    static constexpr const char *kGreen = "\033[32m";
    static constexpr const char *kRed = "\033[31m";
    static constexpr const char *kReset = "\033[0m";

    void pass(std::string_view name)
    {
        ++total;
        ++passed;
        std::cout << std::format("{}[PASS]{} {}\n", kGreen, kReset, name);
    }

    void fail(std::string_view name, std::string_view details)
    {
        ++total;
        std::cout << std::format("{}[FAIL]{} {}: {}\n", kRed, kReset, name, details);
    }

    [[nodiscard]] bool ok() const
    {
        return total == passed;
    }

    int finish() const
    {
        const int failed = total - passed;
        const char *color = failed == 0 ? kGreen : kRed;
        std::cout << std::format("{}\n{} / {} tests passed ({}){}\n", color, passed, total, failed, kReset);
        return failed;
    }
};

#ifndef UNSTRUCTURED_SQUARE_MESH_DIR
#error "UNSTRUCTURED_SQUARE_MESH_DIR is not defined: this should be defined by the build system"
#endif

static inline cuddh::Mesh2D load_unstructured_square()
{
    std::string dir = UNSTRUCTURED_SQUARE_MESH_DIR;
    std::ifstream info(dir + "/info.txt");
    cuddh_verify(info, { std::cerr << std::format("load_unstructured_square error: cannot open {}/info.txt\n", dir); });

    int n_pts, n_elem;
    info >> n_pts >> n_elem;
    info.close();

    cuddh::dmat x(2, n_pts);
    cuddh::imat elems(4, n_elem);

    std::ifstream coo(dir + "/coordinates.txt");
    cuddh_verify(
        coo, { std::cerr << std::format("load_unstructured_square error: cannot open {}/coordinates.txt\n", dir); });

    for (int i = 0; i < n_pts; ++i)
    {
        coo >> x(0, i) >> x(1, i);
    }
    coo.close();

    std::ifstream elements(dir + "/elements.txt");
    cuddh_verify(elements,
                 { std::cerr << std::format("load_unstructured_square error: cannot open {}/elements.txt\n", dir); });

    for (int i = 0; i < n_elem; ++i)
    {
        elements >> elems(0, i) >> elems(1, i) >> elems(2, i) >> elems(3, i);
    }
    elements.close();

    return cuddh::Mesh2D::from_vertices(n_pts, x.data(), n_elem, elems.data());
}
