#pragma once

#include <chrono>
#include <format>
#include <iostream>

#include "Operator.hpp"
#include "Tensor.hpp"
#include "cuddh_config.hpp"
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

    struct SolverParams
    {
        enum Verbosity
        {
            Silent,
            ProgressBar,
            Iteration
        };

        int maxit = 100;    // maximum number of iterations
        double rtol = 1e-3; // relative tolerance for an acceptable solution. Solvers stop when |A*x-b|/|b| < tol.
        double atol = 0.0;  // absolute tolerance for an acceptable solution. Solvers stop when |A*x-b| < atol.
        Verbosity verbose = Silent; // 0: silent, 1: progress bar, 2: one line per iteration
    };

    template <typename scalar_t>
    class Solver
    {
    public:
        Solver(int n) : _n{n} {}
        virtual ~Solver() = default;

        constexpr int ndof() const { return _n; }

        virtual SolverResults solve(scalar_t *x, const scalar_t *b, SolverParams opts = {}) const = 0;

    protected:
        constexpr void set_size(int n) { _n = n; }

    private:
        int _n;
    };

    inline void validate_params(const SolverParams &opts)
    {
        cuddh_verify(opts.maxit > 0, printf("solver error: maxit = %d must be positive\n", opts.maxit));
        cuddh_verify(opts.rtol >= 0, printf("solver error: rtol = %f must be non-negative\n", opts.rtol));
        cuddh_verify(opts.atol >= 0, printf("solver error: atol = %f must be non-negative\n", opts.atol));
    }

    class Timer
    {
    public:
        Timer() : start_time(std::chrono::high_resolution_clock::now()) {}

        void start() { start_time = std::chrono::high_resolution_clock::now(); }

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
        SolverLogger(SolverParams::Verbosity verbosity, int maxit);

        constexpr void log_matvec() { results.num_matvec++; }

        constexpr int num_iterations() const { return results.num_iter; }

        void log_iteration(double res_norm);

        SolverResults log_summary(double res_norm, bool success);

    private:
        int maxit;
        SolverParams::Verbosity verbosity;
        SolverResults results;
        Timer timer;
    };
} // namespace cuddh
