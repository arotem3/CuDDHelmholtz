#include "Operators3D/StiffnessMatrix3D.hpp"

#include "forall.hpp"
#include "linalg.hpp"

using namespace cuddh;

static void setup_geometric_factors(int n_elem, const QuadratureRule &quad, const DeviceMesh3D &mesh, double3x3 *d_G)
{
    int nx = quad.size();

    auto w = quad.w(MemorySpace::DEVICE);
    auto x = quad.x(MemorySpace::DEVICE);

    auto G = reshape(d_G, nx, nx, nx, n_elem);

    forall_3d(nx, nx, nx, n_elem, [=] __device__(int el) mutable -> void {
        const int &i = threadIdx.x;
        const int &j = threadIdx.y;
        const int &k = threadIdx.z;

        __shared__ HexElement elem;
        if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0)
            elem = mesh.element(el);
        __syncthreads();

        const double3 r{x(i), x(j), x(k)};

        auto J = elem.jacobian(r);

        double s = w(i) * w(j) * w(k) / det(J);

        J = adjugate(J);

        double3x3 g;

        for (int m = 0; m < 3; ++m)
        {
            for (int n = 0; n < m; ++n)
            {
                g(m, n) = 0.0;
                for (int l = 0; l < 3; ++l)
                    g(m, n) += J(m, l) * J(n, l);
                g(m, n) *= s;
                g(n, m) = g(m, n);
            }

            g(m, m) = 0.0;
            for (int l = 0; l < 3; ++l)
                g(m, m) += J(m, l) * J(m, l);
            g(m, m) *= s;
        }

        G(i, j, k, el) = g;
    });
}

StiffnessMatrix3D::StiffnessMatrix3D(const H1Space3D &fem) : Operator<double>(fem.size()), fem{fem}
{
    const int n_elem = fem.mesh().n_elem();
    const int n_basis = fem.basis().size();

    auto D = fem.basis().derivative_matrix();

    _D.resize(n_basis * n_basis);
    auto h_D = _D.host_write();
    std::copy(D.data(), D.data() + D.size(), h_D);

    _G.resize(n_basis * n_basis * n_basis * n_elem);

    setup_geometric_factors(n_elem, fem.basis().quadrature(), fem.mesh().to_device(), _G.device_write());
}

template <int NX>
static void stiffness_action(int n_elem, const double *d_D, const double3x3 *d_G, const int *d_I, double c,
                             const double *d_u, double *d_out)
{
    auto D = reshape(d_D, NX, NX);
    auto I = reshape(d_I, NX, NX, NX, n_elem);
    auto G = reshape(d_G, NX, NX, NX, n_elem);

    forall_3d(NX, NX, NX, n_elem, [=] __device__(int el) -> void {
        __shared__ double s_u[NX][NX][NX];
        __shared__ double3 s_F[NX][NX][NX];
        __shared__ double s_D[NX][NX];

        const int &i = threadIdx.x;
        const int &j = threadIdx.y;
        const int &k = threadIdx.z;

        if (k == 0)
        {
            s_D[i][j] = D(i, j);
        }

        const int idx = I(i, j, k, el);
        s_u[i][j][k] = d_u[idx];
        __syncthreads();

        // compute gradient
        double3 grad{0.0, 0.0, 0.0};
        for (int l = 0; l < NX; ++l)
        {
            grad.x += s_D[i][l] * s_u[l][j][k];
            grad.y += s_D[j][l] * s_u[i][l][k];
            grad.z += s_D[k][l] * s_u[i][j][l];
        }
        s_F[i][j][k] = G(i, j, k, el) * grad;
        __syncthreads();

        // integrate against gradient of test function
        double Su = 0.0;
        for (int l = 0; l < NX; ++l)
        {
            Su += s_D[l][i] * s_F[l][j][k].x + s_D[l][j] * s_F[i][l][k].y + s_D[l][k] * s_F[i][j][l].z;
        }
        Su *= c;

        atomicAdd(d_out + idx, Su);
    });
}

void StiffnessMatrix3D::action(double c, const double *x, double *y) const
{
    const double *d_D = _D.device_read();
    const double3x3 *d_G = _G.device_read();
    const int *d_I = fem.global_indices(MemorySpace::DEVICE);

    const int n_elem = fem.mesh().n_elem();
    const int n_basis = fem.basis().size();

    switch (n_basis)
    {
        case 2:
            stiffness_action<2>(n_elem, d_D, d_G, d_I, c, x, y);
            break;
        case 3:
            stiffness_action<3>(n_elem, d_D, d_G, d_I, c, x, y);
            break;
        case 4:
            stiffness_action<4>(n_elem, d_D, d_G, d_I, c, x, y);
            break;
        case 5:
            stiffness_action<5>(n_elem, d_D, d_G, d_I, c, x, y);
            break;
        case 6:
            stiffness_action<6>(n_elem, d_D, d_G, d_I, c, x, y);
            break;
        case 7:
            stiffness_action<7>(n_elem, d_D, d_G, d_I, c, x, y);
            break;
        case 8:
            stiffness_action<8>(n_elem, d_D, d_G, d_I, c, x, y);
            break;
        default:
            cuddh_verify(false, printf("StiffnessMatrix3D::action does not support basis functions of order > 12.\n"));
            break;
    }
}

void StiffnessMatrix3D::action(const double *x, double *y) const
{
    dla::zeros(fem.size(), y);
    action(1.0, x, y);
}

bool StiffnessMatrix3D::assemble(double c, SparseMatrix<double> &S_out) const
{
    const int nb = fem.basis().size();
    const int ne = fem.mesh().n_elem();
    const double *D = _D.host_read();
    const double3x3 *G = _G.host_read();
    auto I = fem.global_indices(MemorySpace::HOST);
    auto Dv = reshape(D, nb, nb);
    auto Gv = reshape(G, nb, nb, nb, ne);

    for (int el = 0; el < ne; ++el)
        for (int a = 0; a < nb; ++a)
            for (int b = 0; b < nb; ++b)
                for (int cv = 0; cv < nb; ++cv)
                {
                    const int row = I(a, b, cv, el);
                    for (int d = 0; d < nb; ++d)
                        for (int e = 0; e < nb; ++e)
                            for (int f = 0; f < nb; ++f)
                            {
                                const int col = I(d, e, f, el);
                                double val = 0.0;
                                if (b == e && cv == f)
                                    for (int l = 0; l < nb; ++l)
                                        val += Dv(l, a) * Gv(l, b, cv, el)(0, 0) * Dv(l, d);
                                if (cv == f)
                                    val += Dv(d, a) * Gv(d, b, cv, el)(1, 0) * Dv(b, e);
                                if (b == e)
                                    val += Dv(d, a) * Gv(d, b, cv, el)(2, 0) * Dv(cv, f);
                                if (cv == f)
                                    val += Dv(a, d) * Gv(a, e, cv, el)(0, 1) * Dv(e, b);
                                if (a == d && cv == f)
                                    for (int l = 0; l < nb; ++l)
                                        val += Dv(l, b) * Gv(a, l, cv, el)(1, 1) * Dv(l, e);
                                if (a == d)
                                    val += Dv(e, b) * Gv(a, e, cv, el)(2, 1) * Dv(cv, f);
                                if (b == e)
                                    val += Dv(a, d) * Gv(a, b, f, el)(0, 2) * Dv(f, cv);
                                if (a == d)
                                    val += Dv(b, e) * Gv(a, b, f, el)(1, 2) * Dv(f, cv);
                                if (a == d && b == e)
                                    for (int l = 0; l < nb; ++l)
                                        val += Dv(l, cv) * Gv(a, b, l, el)(2, 2) * Dv(l, f);
                                S_out.set_value(row, col, c * val);
                            }
                }
    return true;
}

bool StiffnessMatrix3D::assemble(std::complex<double> c, SparseMatrix<double, true> &S_out) const
{
    const int nb = fem.basis().size();
    const int ne = fem.mesh().n_elem();
    const double *D = _D.host_read();
    const double3x3 *G = _G.host_read();
    auto I = fem.global_indices(MemorySpace::HOST);
    auto Dv = reshape(D, nb, nb);
    auto Gv = reshape(G, nb, nb, nb, ne);
    for (int el = 0; el < ne; ++el)
        for (int a = 0; a < nb; ++a)
            for (int b = 0; b < nb; ++b)
                for (int cv = 0; cv < nb; ++cv)
                {
                    const int row = I(a, b, cv, el);
                    for (int d = 0; d < nb; ++d)
                        for (int e = 0; e < nb; ++e)
                            for (int f = 0; f < nb; ++f)
                            {
                                const int col = I(d, e, f, el);
                                double val = 0.0;
                                if (b == e && cv == f)
                                    for (int l = 0; l < nb; ++l)
                                        val += Dv(l, a) * Gv(l, b, cv, el)(0, 0) * Dv(l, d);
                                if (cv == f)
                                    val += Dv(d, a) * Gv(d, b, cv, el)(1, 0) * Dv(b, e);
                                if (b == e)
                                    val += Dv(d, a) * Gv(d, b, cv, el)(2, 0) * Dv(cv, f);
                                if (cv == f)
                                    val += Dv(a, d) * Gv(a, e, cv, el)(0, 1) * Dv(e, b);
                                if (a == d && cv == f)
                                    for (int l = 0; l < nb; ++l)
                                        val += Dv(l, b) * Gv(a, l, cv, el)(1, 1) * Dv(l, e);
                                if (a == d)
                                    val += Dv(e, b) * Gv(a, e, cv, el)(2, 1) * Dv(cv, f);
                                if (b == e)
                                    val += Dv(a, d) * Gv(a, b, f, el)(0, 2) * Dv(f, cv);
                                if (a == d)
                                    val += Dv(b, e) * Gv(a, b, f, el)(1, 2) * Dv(f, cv);
                                if (a == d && b == e)
                                    for (int l = 0; l < nb; ++l)
                                        val += Dv(l, cv) * Gv(a, b, l, el)(2, 2) * Dv(l, f);
                                S_out.set_value(row, col, c * val);
                            }
                }
    return true;
}
