#ifndef CUDDH_OPERATOR_HPP
#define CUDDH_OPERATOR_HPP

namespace cuddh
{
    template <typename scalar_t>
    class Operator
    {
    public:
        Operator(int n) : _n{n} {}
        virtual ~Operator() = default;

        /// @brief y <- y + c * A * x
        virtual void action(scalar_t c, const scalar_t *x, scalar_t *y) const = 0;

        /// @brief y <- A * x
        virtual void action(const scalar_t *x, scalar_t *y) const = 0;

        constexpr int ndof() const { return _n; }

    protected:
        constexpr void set_size(int n) { _n = n; }

    private:
        int _n;
    };
} // namespace cuddh

#endif