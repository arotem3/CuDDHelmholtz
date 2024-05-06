#ifndef CUDDH_HELMHOLTZ_HPP
#define CUDDH_HELMHOLTZ_HPP

#include "cuddh.hpp"

using namespace cuddh;

/// @brief FEM discretization of the Helmholtz equation -div(grad u) - omega^2 u == f
/// with boundary conditions: du/dn + i omega u == 0.
class Helmholtz : public Operator
{
public:
    Helmholtz(double omega_, const double * a2x, const double * ax, const H1Space& fem_, const FaceSpace& fs_)
        : omega{omega_},
          ndof{fem_.size()},
          fdof{fs_.size()},
          fem{fem_},
          fs{fs_},
          S(fem_),
          M(a2x, fem_),
          H(ax, fs_),
          xf(fdof),
          yf(fdof) {}

    /// @brief y[i] = a(x, phi[i]) where a(u,v) = (grad u, grad v) - omega^2 (u, v) - i*omega <u, v> 
    /// @param x the real and imaginary part of the solution
    /// @param y on exit y[i] <- a(x, v[i]) for each v
    void action(const double * x, double * y) const
    {
        const double * u = x;
        const double * v = x + ndof;

        double * Au = y;
        double * Av = y + ndof;

        double * d_yf = yf.device_write();
        double * d_xf = xf.device_write();

        S.action(u, Au);
        S.action(v, Av);

        M.action(-omega * omega, u, Au);
        M.action(-omega * omega, v, Av);

        zeros(fdof, d_yf);
        fs.restrict(v, d_xf);
        H.action(-omega, d_xf, d_yf);
        fs.prolong(d_yf, Au);

        zeros(fdof, d_yf);
        fs.restrict(u, d_xf);
        H.action(omega, d_xf, d_yf);
        fs.prolong(d_yf, Av);
    }

    /// @brief y[i] <- y[i] + c * a(x, phi[i]) where a(u,v) = (grad u, grad v) - omega^2 (u, v) - i*omega <u, v> 
    /// @param c
    /// @param x the real and imaginary part of the solution
    /// @param y on exit y[i] <- a(x, v[i]) for each v
    void action(double c, const double * x, double * y) const
    {
        cuddh_error("Helmholtz::action(c, x, y) not implemented");
    }

private:
    const double omega;
    const int ndof;
    const int fdof;
    
    const H1Space& fem;
    const FaceSpace& fs;
    
    StiffnessMatrix S;
    MassMatrix M;
    FaceMassMatrix H;
    
    mutable host_device_dvec xf;
    mutable host_device_dvec yf;
};

#endif