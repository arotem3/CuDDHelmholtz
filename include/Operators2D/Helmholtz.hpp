#pragma once

#include "FEM2D/GridFunc2D.hpp"
#include "FEM2D/H1Space2D.hpp"
#include "Operators2D/FaceMassMatrix.hpp"
#include "Operators2D/MassMatrix.hpp"
#include "Operators2D/StiffnessMatrix.hpp"
#include "cuddh_error.hpp"

namespace cuddh
{
    /// @brief FEM discretization of the Helmholtz equation -div(grad u) - a(x)^2*omega^2*u == f
    /// with Robin boundary conditions: du/dn - i*omega*a(x)*u == 0.
    ///
    /// Let A = S - omega^2*M - i*omega*H (the n×n complex PDE operator), where S is the
    /// stiffness matrix, M is the mass matrix weighted by a^2, and H is the boundary mass
    /// matrix weighted by a. Complex vectors are stored in blocked format [x_re; x_im].
    ///
    /// **Conjugation convention — must be respected by solvers:**
    /// `action(x, y)` computes the complex CONJUGATE of the standard matvec:
    ///   [y_re; y_im] = [Re(A*z); -Im(A*z)]   where z = x_re + i*x_im
    /// This equals conj(A*z) in the blocked representation.
    ///
    /// `assemble(c, S)` and `SparseMatrix::action` both use the STANDARD (non-conjugated) product.
    ///
    /// To solve  action(x) = b  via a direct factorization of A:
    ///   solve A*y = conj(b),  then x = conj(y)
    /// This is valid because A has real coefficients, so conj(A*z) = A*conj(z).
    class Helmholtz : public Operator<double>
    {
    public:
        Helmholtz(const H1Space2D &fem, const TraceSpace2D &fs, double omega);
        Helmholtz(const H1Space2D &fem, const TraceSpace2D &fs, double omega, const GridFunc2D<double> &a);

        /// @brief Computes [Re(A*z); -Im(A*z)] = conj(A*z) where z = x_re + i*x_im.
        /// @param x blocked device input [x_re; x_im], size 2n
        /// @param y blocked device output [Re(A*z); -Im(A*z)], size 2n
        void action(const double *x, double *y) const;

        void action(double, const double *, double *) const
        {
            cuddh_verify(false, printf("Helmholtz::action(c, x, y) not implemented\n"));
        }

        /// @brief Assembles c*A into the n×n complex sparse matrix S.
        /// S.action uses the standard (non-conjugated) product; see class documentation.
        bool assemble(std::complex<double> c, SparseMatrix<double, true> &S) const override;

    private:
        const double omega;

        StiffnessMatrix S;
        MassMatrix M;
        FaceMassMatrix H;
    };
} // namespace cuddh
