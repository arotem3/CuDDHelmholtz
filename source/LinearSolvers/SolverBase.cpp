#include "LinearSolvers/SolverBase.hpp"

using namespace cuddh;

static std::string format_time(double t)
{
    if (t < 1e-3)
        return std::format("{:.2f}µs", 1e6 * t);
    else if (t < 1.0)
        return std::format("{:.2f}ms", 1e3 * t);
    else if (t < 60.0)
        return std::format("{:.2f}s", t);
    else if (t < 3600.0)
    {
        int minutes = static_cast<int>(t) / 60;
        int seconds = static_cast<int>(t) % 60;
        return std::format("{:02d}m {:02d}s", minutes, seconds);
    }
    else
    {
        int hours = static_cast<int>(t) / 3600;
        int minutes = (static_cast<int>(t) % 3600) / 60;
        int seconds = static_cast<int>(t) % 60;
        return std::format("{:02d}h {:02d}m {:02d}s", hours, minutes, seconds);
    }
}

static std::string progress_bar(int it, int maxit, int len = 30)
{
    int filled = (it * len) / maxit;
    std::string bar(filled, '#');
    std::string empty(len - filled, '-');
    return bar + empty;
}

SolverLogger::SolverLogger(SolverVerbosity verbosity, int maxit) : verbosity(verbosity), maxit(maxit)
{
    results.success = false;
    results.num_iter = 0;
    results.num_matvec = 0;
    results.res_norm.reserve(maxit + 1);
    results.time.reserve(maxit + 1);
}

void SolverLogger::log_iteration(double res_norm)
{
    if (results.num_iter == 0)
    {
        timer.start();
        results.time.push_back(0.0);
    }
    else
    {
        results.time.push_back(timer.elapsed());
    }

    results.res_norm.push_back(res_norm);
    results.num_iter++;

    if (verbosity == SolverVerbosity::ProgressBar)
    {
        std::cout << std::format("\r[{}] || iteration {:10d} / {} || rel. res. = {:10.2e}",
                                 progress_bar(results.num_iter, maxit), results.num_iter, maxit, res_norm)
                  << std::flush;
    }
    else if (verbosity == SolverVerbosity::Iteration)
    {
        std::cout << std::format("iteration {:10d} / {} || rel. res. = {:10.2e}", results.num_iter, maxit, res_norm)
                  << std::endl;
    }
}

SolverResults SolverLogger::log_summary(double res_norm, bool success)
{
    results.success = success;

    if (verbosity != SolverVerbosity::Silent)
    {
        std::cout << std::format("\nAfter {} iterations ({}), solver achieved rel. residual of {:10.2e}",
                                 results.num_iter, format_time(timer.elapsed()), res_norm)
                  << std::endl;
        if (success)
            std::cout << "Solver successfully converged within the maximum number of iterations." << std::endl;
        else
            std::cout << "Solver failed to converge within the maximum number of iterations." << std::endl;
    }

    return std::move(results);
}
