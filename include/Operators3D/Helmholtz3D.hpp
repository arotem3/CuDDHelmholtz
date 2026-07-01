#pragma once

#include "FEM3D/GridFunc3D.hpp"
#include "Operators3D/FaceMassMatrix3D.hpp"
#include "Operators3D/MassMatrix3D.hpp"
#include "Operators3D/StiffnessMatrix3D.hpp"
#include "cuddh_error.hpp"

namespace cuddh
{
    /// @brief FEM discretization of the 3D Helmholtz equation -div(grad u) - a(x)^2*omega^2*u == f
    /// with Robin boundary conditions: du/dn - i*omega*a(x)*u == 0.
    ///
    /// Let A = S - omega^2*M - i*omega*H (the n×n complex PDE operator), where S is the
    /// stiffness matrix, M is the mass matrix weighted by a^2, and H is the boundary mass
    /// matrix weighted by a. Complex vectors are stored in blocked format [x_re; x_im].
    ///
    /// **Conjugation convention:**
    /// `action(x, y)` computes [Re(A*z); -Im(A*z)] = conj(A*z) where z = x_re + i*x_im.
    /// `assemble(c, S)` and `SparseMatrix::action` compute the standard product A*z.
    ///
    /// To solve A*x = b:
    ///   - Iterative (via `action`): pass conj(b) = [b_re; -b_im] as the rhs, since
    ///     action(x) = conj(b)  ⟺  conj(A*x) = conj(b)  ⟺  A*x = b.
    ///   - Direct (via `SparseLU`): assemble A with `assemble`, factorize with `SparseLU`,
    ///     then call `lu.solve(b, x)` directly — no conjugation of b or x needed.
    class Helmholtz3D : public Operator<double>
    {
    public:
        /// @brief Constant-coefficient constructor: a(x) = 1.
        Helmholtz3D(const H1Space3D &fem, const TraceSpace3D &tr, double omega);

        /// @brief Variable-coefficient constructor.
        /// @param a wave-speed coefficient; squared for the mass term, traced for the boundary term.
        Helmholtz3D(const H1Space3D &fem, const TraceSpace3D &tr, double omega, const GridFunc3D<double> &a);

        /// @brief Computes [Re(A*z); -Im(A*z)] = conj(A*z) where z = x_re + i*x_im.
        /// @param x blocked device input [x_re; x_im], size 2n
        /// @param y blocked device output [Re(A*z); -Im(A*z)], size 2n
        void action(const double *x, double *y) const override;

        void action(double, const double *, double *) const override
        {
            cuddh_verify(false, printf("Helmholtz3D::action(c, x, y) not implemented\n"));
        }

        /// @brief Assembles c*A into the n×n complex sparse matrix S.
        /// S.action uses the standard (non-conjugated) product; see class documentation.
        bool assemble(std::complex<double> c, SparseMatrix<double, true> &S) const override;

    private:
        const double omega;

        StiffnessMatrix3D S;
        MassMatrix3D M;
        FaceMassMatrix3D H;
    };
} // namespace cuddh
