#include "LinearFunctional.hpp"

namespace cuddh
{
    LinearFunctional::LinearFunctional(const H1Space& fem_)
        : fem{fem_},
          n_elem{fem.mesh().n_elem()},
          n_basis{fem.basis().size()},
          n_quad{n_basis},
          fast{true},
          metrics{fem.mesh().element_metrics(fem.basis().quadrature())},
          _w(n_quad)
{
    auto& quad = fem.basis().quadrature();

    double * h_w = _w.host_write();
    for (int i = 0; i < n_quad; ++i)
        h_w[i] = quad.w(i);
}

    LinearFunctional::LinearFunctional(const H1Space& fem_, const QuadratureRule& quad)
        : fem{fem_},
          n_elem{fem.mesh().n_elem()},
          n_basis{fem.basis().size()},
          n_quad{quad.size()},
          fast{false},
          metrics{fem.mesh().element_metrics(quad)},
          _w(n_quad),
          _P(n_quad * n_basis)
    {
        fem.basis().eval(n_quad, quad.x(), _P.host_write());

        double * h_w = _w.host_write();
        for (int i = 0; i < n_quad; ++i)
            h_w[i] = quad.w(i);
    }
} // namespace cuddh
