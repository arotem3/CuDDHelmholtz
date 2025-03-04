#ifndef CUDDH_EXAMPLES_HPP
#define CUDDH_EXAMPLES_HPP

#include <iostream>
#include <iomanip>
#include <fstream>
// #include <format>

namespace cuddh
{
    inline static bool to_file(const std::string& fname, int n_dof, const double * u)
    {
        std::ofstream out(fname, std::ios::out | std::ios::binary);
        if (!out.is_open())
        {
            std::cerr << "Failed to open file " << fname << std::endl;
            return false;
        }
        out.write(reinterpret_cast<const char*>(u), n_dof * sizeof(double));
        out.close();

        return true;
    }
} // namespace cuddh


#endif