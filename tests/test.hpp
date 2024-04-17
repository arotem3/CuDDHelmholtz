#ifndef CUDDH_TEST_HPP
#define CUDDH_TEST_HPP

#include "cuddh.hpp"

#include <iostream>
#include <iomanip>
#include <fstream>

namespace cuddh_test
{
    void t_quadrature_rule(int& n_test, int& n_passed);
    void t_basis(int& n_test, int& n_passed);
    void t_gmres(int& n_test, int& n_passed);
    void t_stiffness(int& n_test, int& n_passed);
    void t_mass(int& n_test, int& n_passed);

    cuddh::Mesh2D load_unstructured_square();
} // namespace cuddh_test

#endif
