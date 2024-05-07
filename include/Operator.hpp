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

    class SinglePrecisionOperator
    {
    public:
        SinglePrecisionOperator() = default;
        virtual ~SinglePrecisionOperator() = default;

        virtual void action(const float * x, float * y) const = 0;
    };
} // namespace cuddh


#endif