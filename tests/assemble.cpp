#include <random>

#include "test_common.hpp"

using namespace cuddh;

static std::vector<double> random_vec(int n, unsigned seed = 42)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::vector<double> v(n);
    for (auto &x : v)
        x = dist(rng);
    return v;
}

// Check that sparse matrix assembly matches the matrix-free action.
static bool check_assembly(const Operator<double> &op, const SparseMatrix<double> &sp, const char *label,
                           TestLogger &log, double tol = 1e-10)
{
    const int n = op.ndof();
    auto x_host = random_vec(n);

    // Matrix-free: device path
    host_device_dvec _x(n), _y_mf(n);
    cudaMemcpy(_x.device_write(), x_host.data(), n * sizeof(double), cudaMemcpyHostToDevice);
    dla::zeros(n, _y_mf.device_write());
    op.action(_x.device_read(), _y_mf.device_read_write());
    const double *d_y_mf = _y_mf.device_read();

    std::vector<double> y_mf(n);
    cudaMemcpy(y_mf.data(), d_y_mf, n * sizeof(double), cudaMemcpyDeviceToHost);

    // Sparse: device path
    host_device_dvec _y_sp(n);
    dla::zeros(n, _y_sp.device_write());
    sp.action(_x.device_read(), _y_sp.device_read_write());
    std::vector<double> y_sp(n);
    cudaMemcpy(y_sp.data(), _y_sp.device_read(), n * sizeof(double), cudaMemcpyDeviceToHost);

    double err = 0.0, norm = 0.0;
    for (int i = 0; i < n; ++i)
    {
        err += (y_mf[i] - y_sp[i]) * (y_mf[i] - y_sp[i]);
        norm += y_mf[i] * y_mf[i];
    }
    double rel = (norm > 0.0) ? std::sqrt(err / norm) : std::sqrt(err);

    if (rel < tol)
    {
        log.pass(label);
        return true;
    }
    else
    {
        char buf[256];
        std::snprintf(buf, sizeof(buf), "rel error %.2e exceeds tol %.2e", rel, tol);
        log.fail(label, buf);
        return false;
    }
}

// Check that the assembled complex Helmholtz matrix A matches the matrix-free action.
// Helmholtz::action computes conj(A*z) in blocked format [Re(A*z); -Im(A*z)], while
// SparseMatrix<double,true>::action computes the standard product A*z = [Re(A*z); Im(A*z)].
// So the matrix-free result should equal the sparse result with its imaginary block negated.
static bool check_helmholtz_assembly(const Operator<double> &op, const SparseMatrix<double, true> &sp,
                                     const char *label, TestLogger &log, double tol = 1e-10)
{
    const int n2 = op.ndof();
    const int n = n2 / 2;
    auto x_host = random_vec(n2);

    // Matrix-free: device path. y_mf = conj(A*z) = [Re(A*z); -Im(A*z)]
    host_device_dvec _x(n2), _y_mf(n2);
    cudaMemcpy(_x.device_write(), x_host.data(), n2 * sizeof(double), cudaMemcpyHostToDevice);
    dla::zeros(n2, _y_mf.device_write());
    op.action(_x.device_read(), _y_mf.device_read_write());

    std::vector<double> y_mf(n2);
    cudaMemcpy(y_mf.data(), _y_mf.device_read(), n2 * sizeof(double), cudaMemcpyDeviceToHost);

    // Sparse: device path. y_sp = A*z = [Re(A*z); Im(A*z)]
    host_device_dvec _y_sp(n2);
    dla::zeros(n2, _y_sp.device_write());
    sp.action(_x.device_read(), _y_sp.device_read_write());
    std::vector<double> y_sp(n2);
    cudaMemcpy(y_sp.data(), _y_sp.device_read(), n2 * sizeof(double), cudaMemcpyDeviceToHost);

    double err = 0.0, norm = 0.0;
    for (int i = 0; i < n; ++i)
    {
        double dre = y_mf[i] - y_sp[i];
        double dim = y_mf[n + i] - (-y_sp[n + i]);
        err += dre * dre + dim * dim;
        norm += y_mf[i] * y_mf[i] + y_mf[n + i] * y_mf[n + i];
    }
    double rel = (norm > 0.0) ? std::sqrt(err / norm) : std::sqrt(err);

    if (rel < tol)
    {
        log.pass(label);
        return true;
    }
    else
    {
        char buf[256];
        std::snprintf(buf, sizeof(buf), "rel error %.2e exceeds tol %.2e", rel, tol);
        log.fail(label, buf);
        return false;
    }
}

static void test_2d(TestLogger &log)
{
    const int nx = 6;
    Mesh2D mesh = Mesh2D::uniform_rect(nx, -1.0, 1.0, nx, -1.0, 1.0);
    Basis basis(4);

    H1Space2D fem(mesh, basis);
    ivec bfaces2d = mesh.boundary_edges();
    TraceSpace2D tr(fem, bfaces2d.size(), bfaces2d);
    const int n = fem.size();

    auto make_sparse = [&]() {
        SparseMatrix<double> S(n, n);
        fem.set_pattern(S);
        S.finalize_pattern();
        return S;
    };

    {
        MassMatrix M(fem);
        auto S = make_sparse();
        M.assemble(1.0, S);
        S.finalize_values();
        check_assembly(M, S, "2D MassMatrix assemble", log);
    }
    {
        FaceMassMatrix H(tr);
        auto S = make_sparse();
        H.assemble(1.0, S);
        S.finalize_values();
        check_assembly(H, S, "2D FaceMassMatrix assemble", log);
    }
    {
        StiffnessMatrix K(fem);
        auto S = make_sparse();
        K.assemble(1.0, S);
        S.finalize_values();
        check_assembly(K, S, "2D StiffnessMatrix assemble", log);
    }
    {
        // Helmholtz assembles the complex PDE operator A = S - omega^2*M - i*omega*H.
        // Verify it matches the matrix-free action up to the conjugation convention.
        const double omega = 2.5;
        Helmholtz helm(fem, tr, omega);
        SparseMatrix<double, true> A(n, n);
        fem.set_pattern(A);
        A.finalize_pattern();
        if (helm.assemble(std::complex<double>(1.0, 0.0), A))
        {
            A.finalize_values();
            check_helmholtz_assembly(helm, A, "2D Helmholtz assemble vs conj(action)", log);
        }
        else
            log.fail("2D Helmholtz assemble vs conj(action)", "assemble returned false");
    }
}

static void test_3d(TestLogger &log)
{
    const int nx = 3;
    Mesh3D mesh = Mesh3D::uniform_cube(nx, -1.0, 1.0, nx, -1.0, 1.0, nx, -1.0, 1.0);
    Basis basis(3);

    H1Space3D fem(mesh, basis);
    auto bfaces3d = mesh.get_boundary_faces();
    TraceSpace3D tr(fem, mesh.n_boundary_faces(), bfaces3d.data());
    const int n = fem.size();

    auto make_sparse = [&]() {
        SparseMatrix<double> S(n, n);
        fem.set_pattern(S);
        S.finalize_pattern();
        return S;
    };

    {
        MassMatrix3D M(fem);
        auto S = make_sparse();
        M.assemble(1.0, S);
        S.finalize_values();
        check_assembly(M, S, "3D MassMatrix3D assemble", log);
    }
    {
        FaceMassMatrix3D H(tr);
        auto S = make_sparse();
        H.assemble(1.0, S);
        S.finalize_values();
        check_assembly(H, S, "3D FaceMassMatrix3D assemble", log);
    }
    {
        StiffnessMatrix3D K(fem);
        auto S = make_sparse();
        K.assemble(1.0, S);
        S.finalize_values();
        check_assembly(K, S, "3D StiffnessMatrix3D assemble", log);
    }
    {
        // Helmholtz3D assembles the complex PDE operator A = S - omega^2*M - i*omega*H.
        // Verify it matches the matrix-free action up to the conjugation convention.
        const double omega = 2.5;
        Helmholtz3D helm(fem, tr, omega);
        SparseMatrix<double, true> A(n, n);
        fem.set_pattern(A);
        A.finalize_pattern();
        if (helm.assemble(std::complex<double>(1.0, 0.0), A))
        {
            A.finalize_values();
            check_helmholtz_assembly(helm, A, "3D Helmholtz3D assemble vs conj(action)", log);
        }
        else
            log.fail("3D Helmholtz3D assemble vs conj(action)", "assemble returned false");
    }
}

int main()
{
    TestLogger log;
    test_2d(log);
    test_3d(log);
    return log.finish();
}
