#include "test.hpp"

#include <iostream>
#include <iomanip>

using namespace cuddh_test;

int main()
{
    int num_tests = 0, num_success = 0;

    // t_quadrature_rule(num_tests, num_success);
    // t_basis(num_tests, num_success);
    // t_gmres(num_tests, num_success);
    // t_linalg(num_tests, num_success);
    t_mass(num_tests, num_success);
    // t_stiffness(num_tests, num_success);

    std::cout << std::setw(6) << num_success << " / " << num_tests << " tests passed";
    if (num_tests == num_success)
        std::cout << " :)\n";
    else
        std::cout << " :(\n";

    return 0;
}