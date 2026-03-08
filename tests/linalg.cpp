#include <cmath>
#include <cstdlib>
#include <type_traits>

#include "test_common.hpp"

using namespace cuddh;

template <typename T>
static constexpr const char *type_name()
{
    if constexpr (std::is_same_v<T, float>)
        return "float";
    else
        return "double";
}

template <typename T>
static T tolerance()
{
    if constexpr (std::is_same_v<T, float>)
        return static_cast<T>(1e-5);
    else
        return static_cast<T>(1e-12);
}

template <typename T>
static void test_axpby(TestLogger &summary)
{
    const int n = 1000;

    HostDeviceArray<T> x(n);
    HostDeviceArray<T> y(n);

    T *h_x = x.host_write();
    T *h_y = y.host_write();

    for (int i = 0; i < n; ++i)
    {
        h_x[i] = static_cast<T>(rand()) / static_cast<T>(RAND_MAX) - static_cast<T>(0.2);
        h_y[i] = static_cast<T>(rand()) / static_cast<T>(RAND_MAX) - static_cast<T>(0.2);
    }

    const T *d_x = x.device_read();
    T *d_y = y.device_read_write();

    const T a = static_cast<T>(M_PI);
    const T b = static_cast<T>(M_E);

    dla::axpby(n, a, d_x, b, d_y);

    h_y = y.host_release();
    const T *h_y_result = y.host_read();

    T max_error = static_cast<T>(0.0);
    for (int i = 0; i < n; ++i)
    {
        const T expected = b * h_y[i] + a * h_x[i];
        max_error = std::max(max_error, std::abs(h_y_result[i] - expected));
    }

    const T tol = tolerance<T>();
    if (max_error < tol)
        summary.pass(std::format("linalg axpby ({})", type_name<T>()));
    else
        summary.fail(std::format("linalg axpby ({})", type_name<T>()),
                     std::format("max error {} exceeds tolerance {}", max_error, tol));

    delete[] h_y;
}

template <typename T>
static void test_copy(TestLogger &summary)
{
    const int n = 1 << 10;

    HostDeviceArray<T> x(n);
    HostDeviceArray<T> y(n);

    T *h_x = x.host_write();
    for (int i = 0; i < n; ++i)
        h_x[i] = static_cast<T>(i);

    const T *d_x = x.device_read();
    T *d_y = y.device_write();

    dla::copy(n, d_x, d_y);

    const T *h_y = y.host_read();
    bool is_correct = true;
    for (int i = 0; i < n; ++i)
    {
        if (h_y[i] != static_cast<T>(i))
        {
            is_correct = false;
            break;
        }
    }

    if (is_correct)
        summary.pass(std::format("linalg copy ({})", type_name<T>()));
    else
        summary.fail(std::format("linalg copy ({})", type_name<T>()), "copied vector does not match source");
}

template <typename T>
static void test_dot(TestLogger &summary)
{
    const int n = 1 << 10;

    HostDeviceArray<T> x(n);
    HostDeviceArray<T> y(n);

    T *h_x = x.host_write();
    T *h_y = y.host_write();

    T h_ddot = static_cast<T>(0.0);
    for (int i = 0; i < n; ++i)
    {
        h_x[i] = static_cast<T>(rand()) / static_cast<T>(RAND_MAX) - static_cast<T>(0.2);
        h_y[i] = static_cast<T>(rand()) / static_cast<T>(RAND_MAX) - static_cast<T>(0.2);
        h_ddot += h_x[i] * h_y[i];
    }

    const T *d_x = x.device_read();
    const T *d_y = y.device_read();

    const T ddot_result = dla::dot(n, d_x, d_y);
    const T error = std::abs(h_ddot - ddot_result);

    const T tol = tolerance<T>();
    if (error < tol)
        summary.pass(std::format("linalg dot ({})", type_name<T>()));
    else
        summary.fail(std::format("linalg dot ({})", type_name<T>()),
                     std::format("absolute error {} exceeds tolerance {}", error, tol));
}

template <typename T>
static void test_fill(TestLogger &summary)
{
    const int n = 1 << 10;

    HostDeviceArray<T> x(n);

    const T value = static_cast<T>(M_PI);
    T *d_x = x.device_write();
    dla::fill(n, value, d_x);

    const T *h_x = x.host_read();
    bool is_correct = true;
    for (int i = 0; i < n; ++i)
    {
        if (h_x[i] != value)
        {
            is_correct = false;
            break;
        }
    }

    if (is_correct)
        summary.pass(std::format("linalg fill ({})", type_name<T>()));
    else
        summary.fail(std::format("linalg fill ({})", type_name<T>()), "filled vector does not match constant value");
}

template <typename T>
static void test_scal(TestLogger &summary)
{
    const int n = 1 << 10;

    HostDeviceArray<T> x(n);

    T *h_x = x.host_write();
    for (int i = 0; i < n; ++i)
        h_x[i] = static_cast<T>(rand()) / static_cast<T>(RAND_MAX) - static_cast<T>(0.2);

    const T a = static_cast<T>(M_PI);
    T *d_x = x.device_read_write();
    dla::scal(n, a, d_x);

    h_x = x.host_release();
    const T *h_result = x.host_read();

    const T tol = tolerance<T>() * static_cast<T>(10);
    bool is_correct = true;
    for (int i = 0; i < n; ++i)
    {
        if (std::abs(h_result[i] - a * h_x[i]) > tol)
        {
            is_correct = false;
            break;
        }
    }

    if (is_correct)
        summary.pass(std::format("linalg scal ({})", type_name<T>()));
    else
        summary.fail(std::format("linalg scal ({})", type_name<T>()), "scaled vector does not match expected result");

    delete[] h_x;
}

template <typename T>
static void test_dist(TestLogger &summary)
{
    const int n = 1 << 10;

    HostDeviceArray<T> x(n);
    HostDeviceArray<T> y(n);

    T *h_x = x.host_write();
    T *h_y = y.host_write();

    T h_dist = static_cast<T>(0.0);
    for (int i = 0; i < n; ++i)
    {
        h_x[i] = static_cast<T>(rand()) / static_cast<T>(RAND_MAX) - static_cast<T>(0.2);
        h_y[i] = static_cast<T>(rand()) / static_cast<T>(RAND_MAX) - static_cast<T>(0.2);

        const T e = h_x[i] - h_y[i];
        h_dist += e * e;
    }
    h_dist = std::sqrt(h_dist);

    const T *d_x = x.device_read();
    const T *d_y = y.device_read();

    const T d_dist = dla::dist(n, d_x, d_y);
    const T err = std::abs(d_dist - h_dist) / static_cast<T>(n);

    const T tol = tolerance<T>() * static_cast<T>(100);
    if (err < tol)
        summary.pass(std::format("linalg dist ({})", type_name<T>()));
    else
        summary.fail(std::format("linalg dist ({})", type_name<T>()),
                     std::format("normalized error {} exceeds tolerance {}", err, tol));
}

int main()
{
    TestLogger summary;

    test_axpby<double>(summary);
    test_copy<double>(summary);
    test_dot<double>(summary);
    test_fill<double>(summary);
    test_scal<double>(summary);
    test_dist<double>(summary);

    test_axpby<float>(summary);
    test_copy<float>(summary);
    test_dot<float>(summary);
    test_fill<float>(summary);
    test_scal<float>(summary);
    test_dist<float>(summary);

    return summary.finish();
}
