#include "FaceLinearFunctional.hpp"

namespace cuddh
{
    FaceLinearFunctional::FaceLinearFunctional(const FaceSpace& fs)
        : n_faces{fs.n_faces()},
          n_basis{fs.h1_space().basis().size()},
          quad{fs.h1_space().basis().quadrature()},
          fast{true},
          I{fs.subspace_indices()}
    {
        auto& metrics = fs.metrics(quad);
        detJ = reshape(metrics.measures(), n_basis, n_faces);
        x = reshape(metrics.physical_coordinates(), 2, n_basis, n_faces);
    }

    FaceLinearFunctional::FaceLinearFunctional(const FaceSpace& fs, const QuadratureRule& quad_)
        : n_faces{fs.n_faces()},
          n_basis{fs.h1_space().basis().size()},
          quad{quad_},
          fast{false},
          P(quad.size(), n_basis)
    {
        const int n_quad = quad.size();
        auto& metrics = fs.metrics(quad);
        detJ = reshape(metrics.measures(), n_quad, n_faces);
        x = reshape(metrics.physical_coordinates(), 2, n_quad, n_faces);

        fs.h1_space().basis().eval(n_quad, quad.x(), P);
    }
} // namespace cuddh
