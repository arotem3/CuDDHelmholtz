#pragma once

#include <format>
#include <fstream>
#include <iostream>

namespace cuddh
{
    template <typename T>
    inline static bool to_file(const std::string &fname, int n_dof, const T *u)
    {
        std::ofstream out(fname, std::ios::out | std::ios::binary);
        if (!out.is_open())
        {
            std::cerr << "Failed to open file " << fname << std::endl;
            return false;
        }
        out.write(reinterpret_cast<const char *>(u), n_dof * sizeof(T));
        out.close();

        return true;
    }
} // namespace cuddh
