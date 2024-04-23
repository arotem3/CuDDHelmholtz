#include "FaceLinearFunctional.hpp"

namespace cuddh
{
    FaceLinearFunctional::FaceLinearFunctional(const FaceSpace& fs_)
        : fs{fs_},
          metrics{fs.metrics(fs.h1_space().basis().quadrature())},
          n_faces{fs.n_faces()},
          n_basis{fs.h1_space().basis().size()},
          n_quad{n_basis},
          fast{true},
          _w(n_quad)
    {
        auto& quad = fs.h1_space().basis().quadrature();

        double * h_w = _w.host_write();
        for (int i = 0; i < n_quad; ++i)
            h_w[i] = quad.w(i);
    }

    FaceLinearFunctional::FaceLinearFunctional(const FaceSpace& fs_, const QuadratureRule& quad)
        : fs{fs_},
          metrics{fs.metrics(quad)},
          n_faces{fs.n_faces()},
          n_basis{fs.h1_space().basis().size()},
          n_quad{quad.size()},
          fast{false},
          _w(n_quad),
          _P(n_quad * n_basis)
    {
        double * h_w = _w.host_write();
        for (int i = 0; i < n_quad; ++i)
            h_w[i] = quad.w(i);

        fs.h1_space().basis().eval(n_quad, quad.x(), _P.host_write());
    }
} // namespace cuddh
