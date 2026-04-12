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

    [[nodiscard]] bool ok() const { return total == passed; }

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

/**
 * @brief Test Matrix
 *
 * @details 2D Finite Difference Advection-Diffusion Operator:
 *  -ε Δu + v.∇u = f
 * Square grid: n x n with ndof = n^2 total degrees of freedom
 * Linear indexing: global_index = i*n + j
 */
template <typename scalar_t>
class TestMatrix : public cuddh::Operator<scalar_t>
{
public:
    TestMatrix(int n1d, scalar_t epsilon_, scalar_t vx_, scalar_t vy_)
        : cuddh::Operator<scalar_t>(n1d * n1d), n{n1d}, epsilon{epsilon_}, vx{vx_}, vy{vy_}
    {}

    constexpr ~TestMatrix() = default;

    void action(const scalar_t *x, scalar_t *y) const override
    {
        const scalar_t h = 1.0 / (n - 1);

        cuddh::forall(this->ndof(), [=, *this] __device__(int idx) -> void {
            int i = idx / n;
            int j = idx % n;

            int ip1 = (i + 1) * n + j;
            int im1 = (i - 1) * n + j;
            int jp1 = i * n + (j + 1);
            int jm1 = i * n + (j - 1);

            scalar_t xip1 = (i + 1 < n) ? x[ip1] : scalar_t(0.0);
            scalar_t xim1 = (i - 1 >= 0) ? x[im1] : scalar_t(0.0);
            scalar_t xjp1 = (j + 1 < n) ? x[jp1] : scalar_t(0.0);
            scalar_t xjm1 = (j - 1 >= 0) ? x[jm1] : scalar_t(0.0);

            scalar_t laplacian = (xip1 + xim1 + xjp1 + xjm1 - 4 * x[idx]) / (h * h);

            scalar_t du_dx = (xjp1 - xjm1) / (2 * h);
            scalar_t du_dy = (xip1 - xim1) / (2 * h);
            scalar_t advection = vx * du_dx + vy * du_dy;

            y[idx] = -epsilon * laplacian + advection;
        });
    }

    void action(scalar_t, const scalar_t *, scalar_t *) const override
    {
        cuddh_verify(false, printf("Not Implemented."));
    }

private:
    int n;
    scalar_t epsilon, vx, vy;
};

template <typename scalar_t>
inline auto asym_test_mat(int n = 64)
{
    return TestMatrix<scalar_t>(n, 0.1, 1.0, 0.5);
}

template <typename scalar_t>
inline auto sym_test_mat(int n = 64)
{
    return TestMatrix<scalar_t>(n, 1.0, 0.0, 0.0);
}

template <typename scalar_t>
class InexactPreconditioner : public cuddh::Operator<scalar_t>
{
public:
    InexactPreconditioner(int n_, const cuddh::Operator<scalar_t> &A_) : cuddh::Operator<scalar_t>(n_), A{&A_} {}

    void action(const scalar_t *x, scalar_t *y) const override
    {
        using namespace cuddh;
        dla::zeros(this->ndof(), y);
        gmres(y, *A, x, 5, nullptr, {.maxit = 5, .rtol = 1e-2, .atol = 0.0, .verbose = SolverParams::Silent});
    }

    void action(scalar_t, const scalar_t *, scalar_t *) const override
    {
        cuddh_verify(false, printf("Not Implemented."));
    }

private:
    const cuddh::Operator<scalar_t> *A;
};
