#ifndef CUDDH_EXAMPLES_HPP
#define CUDDH_EXAMPLES_HPP

#include <iostream>
#include <iomanip>
#include <fstream>
// #include <format>

namespace cuddh
{
    inline static void to_file(const std::string& fname, int n_dof, const double * u)
    {
        std::ofstream out(fname, std::ios::out | std::ios::binary);
        out.write(reinterpret_cast<const char*>(u), n_dof * sizeof(double));
        out.close();
    }
} // namespace cuddh


#endif