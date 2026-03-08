#include <cmath>
#include <cstdlib>

#include "test_common.hpp"

using namespace cuddh;

static void test_axpby(TestLogger &summary)
{
    const int n = 1000;

    host_device_dvec x(n);
    host_device_dvec y(n);

    double *h_x = x.host_write();
    double *h_y = y.host_write();

    for (int i = 0; i < n; ++i)
    {
        h_x[i] = static_cast<double>(rand()) / RAND_MAX - 0.2;
        h_y[i] = static_cast<double>(rand()) / RAND_MAX - 0.2;
    }

    const double *d_x = x.device_read();
    double *d_y = y.device_read_write();

    const double a = M_PI;
    const double b = M_E;

    axpby(n, a, d_x, b, d_y);

    h_y = y.host_release();
    const double *h_y_result = y.host_read();

    double max_error = 0.0;
    for (int i = 0; i < n; ++i)
    {
        const double expected = b * h_y[i] + a * h_x[i];
        max_error = std::max(max_error, std::abs(h_y_result[i] - expected));
    }

    if (max_error < 1e-12)
        summary.pass("linalg axpby");
    else
        summary.fail("linalg axpby", std::format("max error {} exceeds tolerance", max_error));

    delete[] h_y;
}

static void test_copy(TestLogger &summary)
{
    const int n = 1 << 10;

    host_device_dvec x(n);
    host_device_dvec y(n);

    double *h_x = x.host_write();
    for (int i = 0; i < n; ++i)
        h_x[i] = static_cast<double>(i);

    const double *d_x = x.device_read();
    double *d_y = y.device_write();

    copy(n, d_x, d_y);

    const double *h_y = y.host_read();
    bool is_correct = true;
    for (int i = 0; i < n; ++i)
    {
        is_correct = h_y[i] == i;
        if (!is_correct)
            break;
    }

    if (is_correct)
        summary.pass("linalg copy");
    else
        summary.fail("linalg copy", "copied vector does not match source");
}

static void test_dot(TestLogger &summary)
{
    const int n = 1 << 10;

    host_device_dvec x(n);
    host_device_dvec y(n);

    double *h_x = x.host_write();
    double *h_y = y.host_write();

    double h_ddot = 0.0;
    for (int i = 0; i < n; ++i)
    {
        h_x[i] = static_cast<double>(rand()) / RAND_MAX - 0.2;
        h_y[i] = static_cast<double>(rand()) / RAND_MAX - 0.2;
        h_ddot += h_x[i] * h_y[i];
    }

    const double *d_x = x.device_read();
    const double *d_y = y.device_read();

    const double ddot_result = dot(n, d_x, d_y);
    const double error = std::abs(h_ddot - ddot_result);

    if (error < 1e-12)
        summary.pass("linalg dot");
    else
        summary.fail("linalg dot", std::format("absolute error {} exceeds tolerance", error));
}

static void test_fill(TestLogger &summary)
{
    const int n = 1 << 10;

    host_device_dvec x(n);

    const double value = M_PI;
    double *d_x = x.device_write();
    fill(n, value, d_x);

    const double *h_x = x.host_read();
    bool is_correct = true;
    for (int i = 0; i < n; ++i)
    {
        is_correct = h_x[i] == value;
        if (!is_correct)
            break;
    }

    if (is_correct)
        summary.pass("linalg fill");
    else
        summary.fail("linalg fill", "filled vector does not match constant value");
}

static void test_scal(TestLogger &summary)
{
    const int n = 1 << 10;

    host_device_dvec x(n);

    double *h_x = x.host_write();
    for (int i = 0; i < n; ++i)
        h_x[i] = static_cast<double>(rand()) / RAND_MAX - 0.2;

    const double a = M_PI;
    double *d_x = x.device_read_write();
    scal(n, a, d_x);

    h_x = x.host_release();
    const double *h_result = x.host_read();

    bool is_correct = true;
    for (int i = 0; i < n; ++i)
    {
        is_correct = h_result[i] == a * h_x[i];
        if (!is_correct)
            break;
    }

    if (is_correct)
        summary.pass("linalg scal");
    else
        summary.fail("linalg scal", "scaled vector does not match expected result");

    delete[] h_x;
}

static void test_dist(TestLogger &summary)
{
    const int n = 1 << 10;

    host_device_dvec x(n);
    host_device_dvec y(n);

    double *h_x = x.host_write();
    double *h_y = y.host_write();

    double h_dist = 0.0;
    for (int i = 0; i < n; ++i)
    {
        h_x[i] = static_cast<double>(rand()) / RAND_MAX - 0.2;
        h_y[i] = static_cast<double>(rand()) / RAND_MAX - 0.2;

        const double e = h_x[i] - h_y[i];
        h_dist += e * e;
    }
    h_dist = std::sqrt(h_dist);

    const double *d_x = x.device_read();
    const double *d_y = y.device_read();

    const double d_dist = dist(n, d_x, d_y);
    const double err = std::abs(d_dist - h_dist) / n;

    if (err < 1e-10)
        summary.pass("linalg dist");
    else
        summary.fail("linalg dist", std::format("normalized error {} exceeds tolerance", err));
}

int main()
{
    TestLogger summary;
    test_axpby(summary);
    test_copy(summary);
    test_dot(summary);
    test_fill(summary);
    test_scal(summary);
    test_dist(summary);
    return summary.finish();
}
