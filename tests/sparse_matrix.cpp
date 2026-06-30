#include "test_common.hpp"

using namespace cuddh;

template <typename scalar_t>
static void test_basic_assembly_and_action(TestLogger &summary, std::string_view name)
{
    SparseMatrix<scalar_t, false> A(3, 3, 5);
    A.add_entry(0, 0);
    A.add_entry(0, 1);
    A.add_entry(1, 1);
    A.add_entry(1, 2);
    A.add_entry(2, 2);

    A.finalize_pattern();
    A.add_entry(0, 0, scalar_t(2));
    A.add_entry(0, 1, scalar_t(-1));
    A.add_entry(1, 1, scalar_t(3));
    A.add_entry(1, 2, scalar_t(4));
    A.add_entry(2, 2, scalar_t(5));

    A.finalize_values();

    scalar_t x[3] = {scalar_t(1), scalar_t(2), scalar_t(3)};
    scalar_t y[3] = {scalar_t(0), scalar_t(0), scalar_t(0)};

    A.action(x, y);

    const scalar_t ref0 = scalar_t(0);  // 2*1 - 1*2
    const scalar_t ref1 = scalar_t(18); // 3*2 + 4*3
    const scalar_t ref2 = scalar_t(15); // 5*3

    const bool ok = (std::abs(y[0] - ref0) < scalar_t(1e-12)) && (std::abs(y[1] - ref1) < scalar_t(1e-12)) &&
                    (std::abs(y[2] - ref2) < scalar_t(1e-12));

    if (ok)
        summary.pass(std::format("Sparse action in finalized state ({})", name));
    else
        summary.fail(std::format("Sparse action in finalized state ({})", name),
                     std::format("got [{}, {}, {}]", y[0], y[1], y[2]));

    scalar_t y2[3] = {scalar_t(0), scalar_t(0), scalar_t(0)};
    A.action(x, y2);

    const bool ok2 = (std::abs(y2[0] - ref0) < scalar_t(1e-12)) && (std::abs(y2[1] - ref1) < scalar_t(1e-12)) &&
                     (std::abs(y2[2] - ref2) < scalar_t(1e-12));

    if (ok2)
        summary.pass(std::format("Sparse action in finalized state ({})", name));
    else
        summary.fail(std::format("Sparse action in finalized state ({})", name),
                     std::format("got [{}, {}, {}]", y2[0], y2[1], y2[2]));
}

static void test_state_transitions(TestLogger &summary)
{
    bool ok = true;

    SparseLU<double> lu = [&ok]() {
        SparseMatrix<double, false> A(2, 2, 2);
        A.add_entry(0, 0);
        A.add_entry(1, 1);

        ok = ok && (A.state() == SparseMatrixState::PatternAssembly);

        A.finalize_pattern();
        ok = ok && (A.state() == SparseMatrixState::COOAssembly);
        A.add_entry(0, 0, 1.0);
        A.add_entry(1, 1, 2.0);

        A.finalize_values();
        ok = ok && (A.state() == SparseMatrixState::Finalized);

        return SparseLU<double, false>(A);
    }();

    const auto &stats = lu.stats();
    ok = ok && (stats.solve_calls == 0);

    double rhs[2] = {1.0, 2.0};
    double sol[2] = {0.0, 0.0};
    ok = ok && lu.solve(rhs, sol);
    ok = ok && (std::abs(sol[0] - 1.0) < 1e-12) && (std::abs(sol[1] - 1.0) < 1e-12);
    ok = ok && (lu.stats().solve_calls == 1);

    double sol2[2] = {0.0, 0.0};
    ok = ok && lu.solve(rhs, sol2);
    ok = ok && (std::abs(sol2[0] - 1.0) < 1e-12) && (std::abs(sol2[1] - 1.0) < 1e-12);
    ok = ok && (lu.stats().solve_calls == 2);

    if (ok)
        summary.pass("Sparse matrix state transitions and SparseLU matrix-lifetime independence");
    else
        summary.fail("Sparse matrix state transitions and SparseLU matrix-lifetime independence",
                     "unexpected state transition result");
}

static void test_complex_assembly(TestLogger &summary)
{
    using cdouble = std::complex<double>;

    SparseMatrix<double, true> A(2, 2, 2);
    A.add_entry(0, 0);
    A.add_entry(1, 1);

    A.finalize_pattern();
    A.add_entry(0, 0, cdouble(1.0, 2.0));
    A.add_entry(1, 1, cdouble(3.0, -1.0));

    A.finalize_values();

    double x[4] = {1.0, 0.0, 0.0, 1.0};
    double y[4] = {0.0, 0.0, 0.0, 0.0};
    A.action(x, y);

    const cdouble y0(y[0], y[2]);
    const cdouble y1(y[1], y[3]);

    const cdouble r0 = cdouble(1.0, 2.0);
    const cdouble r1 = cdouble(1.0, 3.0);

    const bool ok = (std::abs(y0 - r0) < 1e-12) && (std::abs(y1 - r1) < 1e-12);
    const std::string got = std::string("got [") + std::to_string(y0.real()) + ", " + std::to_string(y0.imag()) + "; " +
                            std::to_string(y1.real()) + ", " + std::to_string(y1.imag()) + "]";

    if (ok)
        summary.pass("Sparse complex assembly and action");
    else
        summary.fail("Sparse complex assembly and action", got);
}

static void test_factor_and_solve(TestLogger &summary)
{
    SparseMatrix<double, false> A(2, 2, 2);
    A.add_entry(0, 0);
    A.add_entry(1, 1);
    A.finalize_pattern();
    A.add_entry(0, 0, 2.0);
    A.add_entry(1, 1, 3.0);
    A.finalize_values();

    SparseLU<double, false> lu(A);

    bool ok = (lu.stats().analysis_seconds >= 0.0) && (lu.stats().factor_seconds >= 0.0);
    ok = ok && (lu.stats().finalized_bytes > 0);
    ok = ok && (lu.stats().factor_bytes > 0);

    double rhs[2] = {2.0, 6.0};
    double sol[2] = {0.0, 0.0};
    ok = ok && lu.solve(rhs, sol);
    ok = ok && (std::abs(sol[0] - 1.0) < 1e-12) && (std::abs(sol[1] - 2.0) < 1e-12);
    ok = ok && (lu.stats().solve_calls == 1);
    ok = ok && (lu.stats().total_solve_seconds >= 0.0);

    SparseMatrix<double, true> C(1, 1, 1);
    C.add_entry(0, 0);
    C.finalize_pattern();
    C.add_entry(0, 0, std::complex<double>(2.0, 1.0));
    C.finalize_values();
    SparseLU<double, true> clu(C);
    double crhs[2] = {2.0, 1.0};
    double csol[2] = {0.0, 0.0};
    ok = ok && clu.solve(crhs, csol);
    ok = ok && (std::abs(csol[0] - 1.0) < 1e-12) && (std::abs(csol[1] - 0.0) < 1e-12);
    ok = ok && (clu.stats().solve_calls == 1);

    if (ok)
        summary.pass("SparseLU factor, solve, and stats (real + blocked-complex)");
    else
        summary.fail(
            "SparseLU factor, solve, and stats (real + blocked-complex)",
            std::format("got real [{:.3e}, {:.3e}] and complex [{:.3e}, {:.3e}]", sol[0], sol[1], csol[0], csol[1]));
}

int main()
{
    TestLogger summary;

    test_basic_assembly_and_action<double>(summary, "double");
    test_complex_assembly(summary);
    test_state_transitions(summary);
    test_factor_and_solve(summary);

    return summary.finish();
}
