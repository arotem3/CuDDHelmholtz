#ifndef CUDDH_OPERATOR_HPP
#define CUDDH_OPERATOR_HPP

namespace cuddh
{
    class Operator
    {
    public:
        Operator() = default;
        virtual ~Operator() = default;

        /// @brief y <- y + c * A * x
        virtual void action(double c, const double * x, double * y) const = 0;

        /// @brief y <- A * x 
        virtual void action(const double * x, double * y) const = 0;
    };

    class Identity : public Operator
    {
    public:
        Identity(int n_) : n{n_} {}
        ~Identity() = default;

        void action(double c, const double * x, double * y) const override
        {
            for (int i = 0; i < n; ++i)
                y[i] += c * x[i];
        }

        void action(const double * x, double * y) const override
        {
            for (int i = 0; i < n; ++i)
                y[i] = x[i];
        }

    private:
        const int n;
    };
} // namespace cuddh


#endif