#include "cuddh_error.hpp"

namespace cuddh
{
    void cuddh_error(const std::string& msg)
    {
        throw std::runtime_error(msg);
    }
} // namespace cuddh
