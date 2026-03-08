#ifndef CUDDH_OPERATOR_HPP
#define CUDDH_OPERATOR_HPP

namespace cuddh
{
    template <typename scalar_t>
    class Operator
    {
    public:
        Operator() = default;
        virtual ~Operator() = default;

        /// @brief y <- y + c * A * x
        virtual void action(scalar_t c, const scalar_t *x, scalar_t *y) const = 0;

        /// @brief y <- A * x
        virtual void action(const scalar_t *x, scalar_t *y) const = 0;
    };
} // namespace cuddh

#endif