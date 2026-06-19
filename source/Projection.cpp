#include "Projection.hpp"

namespace cuddh
{
    void polynomial_projection_matrix(MatrixWrapper<double> P, const Basis &basis, const QuadratureRule &quad)
    {
        int nq = quad.size();
        int nb = basis.size();

        cuddh_verify(P.shape(0) == nq && P.shape(1) == nb,
                     printf("polynomial_projection_matrix error: Dimensions of P (%d x %d) do not match quad.size() "
                            "(=%d) x basis.size() (=%d)\n",
                            P.shape(0), P.shape(1), quad.size(), basis.size()));

        basis.eval(nq, quad.x(MemorySpace::HOST), P);

        auto qw = quad.w(MemorySpace::HOST);
        auto bw = basis.quadrature().w(MemorySpace::HOST);

        for (int j = 0; j < nb; ++j)
        {
            for (int i = 0; i < nq; ++i)
            {
                P(i, j) *= qw[i] / bw[j];
            }
        }
    }
} // namespace cuddh
