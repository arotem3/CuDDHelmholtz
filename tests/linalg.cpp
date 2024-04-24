#include "test.hpp"

using namespace cuddh;

namespace cuddh_test
{
    static void t_axpby(int & n_test, int & n_passed)
    {
        const int n = 1000;

        host_device_dvec x(n);
        host_device_dvec y(n);

        double * h_x = x.host_write();
        double * h_y = y.host_write();

        for (int i=0; i < n; ++i)
        {
            h_x[i] = (double)rand() / RAND_MAX - 0.2;
            h_y[i] = (double)rand() / RAND_MAX - 0.2;
        }

        const double * d_x = x.device_read();
        double * d_y = y.device_read_write();

        const double a = M_PI;
        const double b = M_E;

        axpby(n, a, d_x, b, d_y);

        h_y = y.host_release();
        const double * h_y_result = y.host_read();

        double max_error = 0.0;
        for (int i=0; i < n; ++i)
        {
            double c = b * h_y[i] + a * h_x[i];
            max_error = std::max(max_error, std::abs(h_y_result[i] - c));
        }

        n_test++;
        if (max_error < 1e-12)
        {
            std::cout << "\t[ + ] t_axpby() test successful." << std::endl;
            n_passed++;
        }
        else
        {
            std::cout << "\t[ - ] t_axpby() test failed.\n\t\tComputed error ~ " << max_error << "but should have been exact to machine prec." << std::endl;
        }

        delete[] h_y;
    }

    static void t_copy(int& n_test, int& n_passed)
    {
        const int n = 1 << 10;

        host_device_dvec x(n);
        host_device_dvec y(n);

        double * h_x = x.host_write();
        
        for (int i=0; i < n; ++i)
        {
            h_x[i] = (double)i;
        }

        const double * d_x = x.device_read();
        double * d_y = y.device_write();

        copy(n, d_x, d_y);

        const double * h_y = y.host_read();

        bool is_correct = true;
        for (int i = 0; i < n; ++i)
        {
            is_correct = h_y[i] == i;
            if (not is_correct)
                break;
        }

        n_test++;
        if (is_correct)
        {
            std::cout << "\t[ + ] t_copy() test successful." << std::endl;
            n_passed++;
        }
        else
        {
            std::cout << "\t[ - ] t_copy() test failed." << std::endl;
        }
    }

    static void t_dot(int& n_test, int& n_passed)
    {
        const int n = 1000;

        host_device_dvec x(n);
        host_device_dvec y(n);

        double * h_x = x.host_write();
        double * h_y = y.host_write();

        double h_ddot = 0.0;
        for (int i = 0; i < n; ++i)
        {
            h_x[i] = (double)rand() / RAND_MAX - 0.2; // [-0.2, 0.8]
            h_y[i] = (double)rand() / RAND_MAX - 0.2;

            h_ddot += h_x[i] * h_y[i];
        }

        const double * d_x = x.device_read();
        const double * d_y = y.device_read();

        const double ddot_result = dot(n, d_x, d_y);

        const double error = std::abs(h_ddot - ddot_result);

        n_test++;
        if (error < 1e-12)
        {
            std::cout << "\t[ + ] t_dot test successful." << std::endl;
            n_passed++;
        }
        else
        {
            std::cout << "\t[ - ] t_dot test failed.\n\t\tComputed error ~ " << error << "but should have been exact to machine prec." << std::endl;
        }
    }

    static void t_fill(int& n_test, int& n_passed)
    {
        const int n = 1 << 10;

        host_device_dvec x(n);

        double * d_x = x.device_write();
        const double a = M_PI;

        fill(n, a, d_x);

        const double * h_x = x.host_read();

        bool is_correct = true;
        for (int i=0; i < n; ++i)
        {
            is_correct = h_x[i] == a;
            if (not is_correct)
                break;
        }

        n_test++;
        if (is_correct)
        {
            std::cout << "\t[ + ] t_fill() test successful." << std::endl;
            n_passed++;
        }
        else
        {
            std::cout << "\t[ - ] t_fill() test failed." << std::endl;
        }
    }

    static void t_scal(int& n_test, int& n_passed)
    {
        const int n = 1 << 10;

        host_device_dvec x(n);

        double * h_x = x.host_write();

        for (int i = 0; i < n; ++i)
        {
            h_x[i] = (double)rand() / RAND_MAX - 0.2;
        }

        double * d_x = x.device_read_write();

        const double a = M_PI;

        scal(n, a, d_x);

        h_x = x.host_release();
        const double * h_result = x.host_read();

        bool is_correct = true;
        for (int i = 0; i < n; ++i)
        {
            is_correct = h_result[i] == a * h_x[i];
            if (not is_correct)
                break;
        }

        n_test++;
        if (is_correct)
        {
            std::cout << "\t[ + ] t_scal() test successful." << std::endl;
            n_passed++;
        }
        else
        {
            std::cout << "\t[ - ] t_scal() test failed." << std::endl;
        }

        delete[] h_x;
    }

    static void t_dist(int& n_test, int& n_passed)
    {
        const int n = 1<<10;

        host_device_dvec x(n);
        host_device_dvec y(n);

        double * h_x = x.host_write();
        double * h_y = y.host_write();

        double h_dist = 0.0;
        for (int i = 0; i < n; ++i)
        {
            h_x[i] = (double)rand() / RAND_MAX - 0.2; // [-0.2, 0.8]
            h_y[i] = (double)rand() / RAND_MAX - 0.2;

            double e = h_x[i] - h_y[i];
            h_dist += e * e;
        }
        h_dist = std::sqrt(h_dist);

        const double * d_x = x.device_read();
        const double * d_y = y.device_read();

        const double d_dist = dist(n, d_x, d_y);

        const double err = std::abs(d_dist - h_dist) / n;

        n_test++;
        if (err < 1e-10)
        {
            std::cout << "\t[ + ] t_dist() test successful." << std::endl;
            n_passed++;
        }
        else
        {
            std::cout << "\t[ - ] t_dist() test failed.\n\t\tComputed error ~ " << err << "but should have been exact to machine prec." << std::endl;
        }
    }

    void t_linalg(int& n_test, int& n_passed)
    {
        t_axpby(n_test, n_passed);
        t_copy(n_test, n_passed);
        t_dot(n_test, n_passed);
        t_fill(n_test, n_passed);
        t_scal(n_test, n_passed);
        t_dist(n_test, n_passed);
    }
} // namespace cuddh_test
