#include "LinearFunctional.hpp"

namespace cuddh
{
    LinearFunctional::LinearFunctional(const H1Space& fem)
        : n_elem{fem.mesh().n_elem()},
          n_basis{fem.basis().size()},
          quad{fem.basis().quadrature()},
          fast{true},
          I(fem.global_indices(), n_basis, n_basis, n_elem)
    {
        auto& metrics = fem.mesh().element_metrics(quad);
        detJ = reshape(metrics.measures(), n_basis, n_basis, n_elem);
        x = reshape(metrics.physical_coordinates(), 2, n_basis, n_basis, n_elem);
    }

    LinearFunctional::LinearFunctional(const H1Space& fem, const QuadratureRule& quad_)
        : n_elem{fem.mesh().n_elem()},
          n_basis{fem.basis().size()},
          quad{quad_},
          fast{false},
          I(fem.global_indices(), n_basis, n_basis, n_elem),
          P(quad.size(), n_basis)
    {
        const int n_quad = quad.size();

        fem.basis().eval(n_quad, quad.x(), P);
        
        auto& metrics = fem.mesh().element_metrics(quad);
        detJ = reshape(metrics.measures(), n_quad, n_quad, n_elem);
        x = reshape(metrics.physical_coordinates(), 2, n_quad, n_quad, n_elem);
    }
} // namespace cuddh
