#include "FaceMassMatrix.hpp"

namespace cuddh
{
    template <int NQ>
    static void init_face_mass(int n_faces,
                               int n_basis,
                               int n_quad,
                               const double * d_w,
                               const double * d_P,
                               const double * d_detJ,
                               const double * d_a,
                               const int * d_I,
                               double * d_op)
    {
        auto w = reshape(d_w, n_quad);
        auto P = reshape(d_P, n_quad, n_basis);
        auto detJ = reshape(d_detJ, n_quad, n_faces);
        auto I = reshape(d_I, n_basis, n_faces);
        auto op = reshape(d_op, n_quad, n_faces);

        forall_1d(n_quad, n_faces, [=] __device__ (int e) -> void
        {
            __shared__ double a[NQ];

            const int k = threadIdx.x;
            int idx;

            // copy a
            if (k < n_basis)
            {
                idx = I(k, e);
                a[k] = (d_a) ? d_a[idx] : 1.0;
            }

            __sync_threads();

            // eval on quadrature points
            double pa = 0.0;
            for (int l = 0; l < n_basis; ++l)
                pa += P(i, l) * a[l];
            
            // scale by weights and detJ
            pa *= w(k) * detJ(k, e);

            // map to global
            op(k, e) = pa;
        });
    }

    FaceMassMatrix::FaceMassMatrix(const FaceSpace& fs_)
        : fs{fs_},
          ndof{fs.size()},
          n_faces{fs.n_faces()},
          n_basis{fs.h1_space().basis().size()},
          n_quad{fs.h1_space().mesh().max_element_order() + n_basis},
          _a(n_quad * n_faces),
          _P(n_quad * n_basis)
    {
        QuadratureRule quad(n_quad, QuadratureRule::GaussLegendre);

        host_device_dvec _w(n_quad);
        double * h_w = _w.host_write();
        for (int i = 0; i < n_quad; ++i)
            h_w = quad.w(i);

        auto& basis = fs.h1_space().basis();
        basis.eval(n_quad, quad.x(), _P.host_write());

        auto& metrics = fs.metrics(quad);

        const double * d_w = _w.device_read();
        const double * d_P = _P.device_read();
        const double * d_detJ = metrics.measures(MemorySpace::DEVICE);
        const int * d_I = fs.subspace_indices(MemorySpace::DEVICE);
        double * d_op = _a.device_write();

        if (n_quad <= 4)
            init_face_mass<4>(n_faces, n_basis, n_quad, d_w, d_P, d_detJ, nullptr, d_I, d_op);
        else if (n_quad <= 8)
            init_face_mass<8>(n_faces, n_basis, n_quad, d_w, d_P, d_detJ, nullptr, d_I, d_op);
        else if (n_quad <= 12)
            init_face_mass<12>(n_faces, n_basis, n_quad, d_w, d_P, d_detJ, nullptr, d_I, d_op);
        else if (n_quad <= 16)
            init_face_mass<16>(n_faces, n_basis, n_quad, d_w, d_P, d_detJ, nullptr, d_I, d_op);
        else if (n_quad <= 24)
            init_face_mass<24>(n_faces, n_basis, n_quad, d_w, d_P, d_detJ, nullptr, d_I, d_op);
        else if (n_quad <= 32)
            init_face_mass<32>(n_faces, n_basis, n_quad, d_w, d_P, d_detJ, nullptr, d_I, d_op);
        else if (n_quad <= 64)
            init_face_mass<64>(n_faces, n_basis, n_quad, d_w, d_P, d_detJ, nullptr, d_I, d_op);
        else
            cuddh_error("FaceMassMatrix does not support quadrature rules with more than 64 points.");
    }

    FaceMassMatrix::FaceMassMatrix(const double * a, const FaceSpace& fs_)
        : fs{fs_},
          ndof{fs.size()},
          n_faces{fs.n_faces()},
          n_basis{fs.h1_space().basis().size()},
          n_quad{fs.h1_space().mesh().max_element_order() + n_basis},
          _a(n_quad * n_faces),
          _P(n_quad * n_basis)
    {
        QuadratureRule quad(n_quad, QuadratureRule::GaussLegendre);

        host_device_dvec _w(n_quad);
        double * h_w = _w.host_write();
        for (int i = 0; i < n_quad; ++i)
            h_w = quad.w(i);

        auto& basis = fs.h1_space().basis();
        basis.eval(n_quad, quad.x(), _P.host_write());

        auto& metrics = fs.metrics(quad);

        const double * d_w = _w.device_read();
        const double * d_P = _P.device_read();
        const double * d_detJ = metrics.measures(MemorySpace::DEVICE);
        const int * d_I = fs.subspace_indices(MemorySpace::DEVICE);
        double * d_op = _a.device_write();

        if (n_quad <= 4)
            init_face_mass<4>(n_faces, n_basis, n_quad, d_w, d_P, d_detJ, a, d_I, d_op);
        else if (n_quad <= 8)
            init_face_mass<8>(n_faces, n_basis, n_quad, d_w, d_P, d_detJ, a, d_I, d_op);
        else if (n_quad <= 12)
            init_face_mass<12>(n_faces, n_basis, n_quad, d_w, d_P, d_detJ, a, d_I, d_op);
        else if (n_quad <= 16)
            init_face_mass<16>(n_faces, n_basis, n_quad, d_w, d_P, d_detJ, a, d_I, d_op);
        else if (n_quad <= 24)
            init_face_mass<24>(n_faces, n_basis, n_quad, d_w, d_P, d_detJ, a, d_I, d_op);
        else if (n_quad <= 32)
            init_face_mass<32>(n_faces, n_basis, n_quad, d_w, d_P, d_detJ, a, d_I, d_op);
        else if (n_quad <= 64)
            init_face_mass<64>(n_faces, n_basis, n_quad, d_w, d_P, d_detJ, a, d_I, d_op);
        else
            cuddh_error("FaceMassMatrix does not support quadrature rules with more than 64 points.");
    }

    template <int NQ>
    static void mass_action(int n_faces,
                            int n_basis,
                            int n_quad,
                            const double * d_P,
                            const double * d_a,
                            const int * d_I,
                            double c,
                            const double * d_U,
                            double * d_out)
    {
        auto P = reshape(d_P, n_quad, n_basis);
        auto a = reshape(d_a, n_quad, n_faces);
        auto I = reshape(d_I, n_basis, n_faces);

        forall_1d(n_quad, n_faces, [=] __device__ (int f) -> void
        {
            __shared__ double u[NQ];
            __shared__ double Pu[NQ];

            const int k = threadIdx.x;
            int idx;

            // copy faces values to u
            if (k < n_basis)
            {
                idx = I(k, f);
                u[k] = d_U[idx];
            }

            __syncthreads();

            // eval on quadrature rule and scale by a(x)*detJ*w
            double pu = 0.0;
            for (int l = 0; l < n_basis; ++l)
                pu += P(k, l) * u[l];
            pu *= a(k, f);
            Pu[k] = pu;

            __syncthreads();

            // integrate
            if (k < n_basis)
            {
                Mu = 0.0;
                for (int i = 0; i < n_quad; ++i)
                    Mu += P(i, k) * Pu[i];
                Mu *= c;

                AtomicAdd(d_out + idx, Mu);
            }
        });
    }

    void FaceMassMatrix::action(double c, const double * x, double * y) const
    {
        const double * d_P = _P.device_read();
        const double * d_a = _a.device_read();
        const int * d_I = fs.subspace_indices(MemorySpace::DEVICE);

        if (n_quad <= 4)
            mass_action<4>(n_faces, n_basis, n_quad, d_P, d_a, d_I, c, x, y);
        else if (n_quad <= 8)
            mass_action<8>(n_faces, n_basis, n_quad, d_P, d_a, d_I, c, x, y);
        else if (n_quad <= 12)
            mass_action<12>(n_faces, n_basis, n_quad, d_P, d_a, d_I, c, x, y);
        else if (n_quad <= 16)
            mass_action<16>(n_faces, n_basis, n_quad, d_P, d_a, d_I, c, x, y);
        else if (n_quad <= 24)
            mass_action<24>(n_faces, n_basis, n_quad, d_P, d_a, d_I, c, x, y);
        else if (n_quad <= 32)
            mass_action<32>(n_faces, n_basis, n_quad, d_P, d_a, d_I, c, x, y);
        else if (n_quad <= 64)
            mass_action<64>(n_faces, n_basis, n_quad, d_P, d_a, d_I, c, x, y);
        else
            cuddh_error("FaceMassMatrix::action does not support quadrature rules with more than 64 points");
    }

    void FaceMassMatrix::action(const double * x, double * y) const
    {
        zeros(ndof, y);
        action(1.0, x, y);
    }

    static void init_diag(int ndof,
                          int n_faces,
                          int n_basis,
                          const double * d_w,
                          const double * d_detJ,
                          const double * d_a,
                          const int * d_I,
                          double * d_op)
    {
        auto w = reshape(d_w, n_basis);
        auto detJ = reshape(d_detJ, n_basis, n_faces);
        auto d_i = reshape(d_I, n_basis, n_faces);

        forall_1d(n_basis, nf, [=] __device__ (int f) -> void
        {
            const int k = threadIdx.x;
            const int idx = I(k, f);

            double a = (d_a) ? d_a[idx] : 1.0;
            a *= w(k) * detJ(k, f);

            AtomicAdd(d_op + idx, a);
        });

        forall(ndof, [=] __device__ (int i) -> void
        {
            d_op[i] = 1.0 / d_op[i];
        });


    }

    DiagInvFaceMassMatrix::DiagInvFaceMassMatrix(const FaceSpace& fs)
        : ndof(fs.size()),
          inv_m(ndof)
    {
        const int nf = fs.n_faces();
        auto& q = fs.h1_space().basis().quadrature();
        const int n_basis = q.size();

        host_device_dvec _w(n_basis);
        double * h_w = _w.host_write();
        for (int i = 0; i < n_basis; ++i)
            h_w = q.w(i);
        const double * d_w = _w.device_read();

        auto& metrics = fs.metrics(q);
        auto d_detJ = metrics.measures(MemorySpace::DEVICE);

        auto d_I = fs.subspace_indices(MemorySpace::DEVICE);
        const double * d_inv_m = inv_m.device_read();

        init_diag(ndof, n_faces, n_basis, d_w, d_detJ, nullptr, d_I, d_inv_m);
    }

    DiagInvFaceMassMatrix::DiagInvFaceMassMatrix(const double * a, const FaceSapce& fs)
        : ndof{fs.size()},
          inv_m(ndf)
    {
        const int nf = fs.n_faces();
        auto& q = fs.h1_space().basis().quadrature();
        const int n_basis = q.size();

        host_device_dvec _w(n_basis);
        double * h_w = _w.host_write();
        for (int i = 0; i < n_basis; ++i)
            h_w = q.w(i);
        const double * d_w = _w.device_read();

        auto& metrics = fs.metrics(q);
        auto d_detJ = metrics.measures(MemorySpace::DEVICE);

        auto d_I = fs.subspace_indices(MemorySpace::DEVICE);
        const double * d_inv_m = inv_m.device_read();

        init_diag(ndof, n_faces, n_basis, d_w, d_detJ, a, d_I, d_inv_m);
    }

    void DiagInvFaceMassMatrix::action(double c, const double * x, double * y) const
    {
        auto d_inv_m = inv_m.device_read();

        forall(ndof, [=] __device__ (int i) -> void
        {
            y[i] += c * d_inv_m[i] * x[i];
        });
    }

    void DiagInvFaceMassMatrix::action(const double * x, double * y) const
    {
        auto d_inv_m = inv_m.device_read();

        forall(ndof, [=] __device__ (int i) -> void
        {
            y[i] = d_inv_m[i] * x[i];
        });
    }
} // namespace cuddh
