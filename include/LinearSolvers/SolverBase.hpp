#pragma once

#include <thrust/device_vector.h>
#include <thrust/universal_vector.h>

#include <chrono>
#include <format>
#include <iomanip>
#include <iostream>

#include "Operator.hpp"
#include "Tensor.hpp"
#include "linalg.hpp"

namespace cuddh
{
    struct SolverResults
    {
        bool success;
        int num_iter;
        int num_matvec;
        std::vector<double> res_norm;
        std::vector<double> time;
    };

    enum class SolverVerbosity
    {
        Silent,
        ProgressBar,
        Iteration
    };

    class Timer
    {
    public:
        Timer() : start_time(std::chrono::high_resolution_clock::now()) {}

        void start()
        {
            start_time = std::chrono::high_resolution_clock::now();
        }

        double elapsed() const
        {
            auto now = std::chrono::high_resolution_clock::now();
            return std::chrono::duration_cast<std::chrono::nanoseconds>(now - start_time).count() * 1e-9;
        }

    private:
        std::chrono::high_resolution_clock::time_point start_time;
    };

    class SolverLogger
    {
    public:
        SolverLogger(SolverVerbosity verbosity, int maxit);

        constexpr void log_matvec()
        {
            results.num_matvec++;
        }

        constexpr int num_iterations() const
        {
            return results.num_iter;
        }

        void log_iteration(double res_norm);

        SolverResults log_summary(double res_norm, double tol);

    private:
        int maxit;
        SolverVerbosity verbosity;
        SolverResults results;
        Timer timer;
    };
} // namespace cuddh
