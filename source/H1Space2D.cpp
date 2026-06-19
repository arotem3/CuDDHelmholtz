#include "FEM2D/H1Space2D.hpp"

namespace cuddh
{
    H1Space2D::H1Space2D(const Mesh2D &mesh_, const Basis &basis_)
        : n_elem{mesh_.n_elem()}, n_basis{basis_.size()}, _mesh{mesh_}, _basis{basis_}, _I(n_basis * n_basis * n_elem)
    {
        icube_wrapper I(_I.host_write(), n_basis, n_basis, n_elem);

        std::unordered_map<int, int> duplicate_dof_map; // maps global DOF index to unique representative DOF index

        // follow mask chain to true canonical representative (not in mask)
        auto canonical = [&duplicate_dof_map](int x) -> int {
            while (duplicate_dof_map.contains(x))
                x = duplicate_dof_map.at(x);
            return x;
        };

        // iterate over interior edges to identify duplicate DOFs
        const int n_edges = _mesh.n_edges(FaceType::INTERIOR);
        for (int e = 0; e < n_edges; ++e)
        {
            EdgeConnectivity edge = _mesh.edge_connectivity(e, FaceType::INTERIOR);

            for (int i = 0; i < n_basis; ++i)
            {
                const int2 v0 = edge2vol(n_basis, i, edge.labels[0]);
                const int idx0 = v0.x + n_basis * (v0.y + n_basis * edge.elements[0]);

                const int j = permute_edge_index(n_basis, i, edge.permutation);
                const int2 v1 = edge2vol(n_basis, j, edge.labels[1]);
                const int idx1 = v1.x + n_basis * (v1.y + n_basis * edge.elements[1]);

                const int c0 = canonical(idx0);
                const int c1 = canonical(idx1);
                if (c0 != c1)
                    duplicate_dof_map.insert({c1, c0});
            }
        }

        const int N = n_elem * n_basis * n_basis;
        ndof = N - duplicate_dof_map.size();
        int l = 0;
        for (int i = 0; i < N; ++i)
        {
            if (not duplicate_dof_map.contains(i))
            {
                I[i] = l;
                ++l;
            }
        }

        for (auto [idx1, idx0] : duplicate_dof_map)
            I[idx1] = I[canonical(idx0)];

        _xy.resize(ndof);
        auto xy = reshape(_xy.host_write(), ndof);

        auto q = _basis.quadrature().x(MemorySpace::HOST);

        for (int el = 0; el < n_elem; ++el)
        {
            const QuadElement elem = _mesh.element(el);

            for (int j = 0; j < n_basis; ++j)
            {
                for (int i = 0; i < n_basis; ++i)
                {
                    const double2 xi{q(i), q(j)};
                    xy(I(i, j, el)) = elem.physical_coordinates(xi);
                }
            }
        }
    }

    TraceSpace2D::TraceSpace2D(const H1Space2D &fem_, int nf, const int *faces_)
        : fem{fem_}, _n_faces{nf}, n_basis{fem.basis().size()}, _I(n_basis * nf), _faces(nf)
    {
        auto F = reshape(_faces.host_write(), nf);
        auto I = reshape(_I.host_write(), n_basis, nf);

        for (int i = 0; i < nf; ++i)
            F(i) = faces_[i];

        const Mesh2D &mesh = fem.mesh();
        const int n_elem = mesh.n_elem();
        auto K = reshape(fem.global_indices(MemorySpace::HOST), n_basis, n_basis, n_elem);

        std::unordered_map<int, int> global_to_trace;
        std::vector<int> P;

        global_to_trace.reserve(n_basis * nf);
        P.reserve(n_basis * nf);

        int l = 0;
        for (int f = 0; f < nf; ++f)
        {
            const EdgeConnectivity edge = mesh.edge_connectivity(F(f));
            const int el = edge.elements[0];

            for (int i = 0; i < n_basis; ++i)
            {
                const auto [x, y] = edge2vol(n_basis, i, edge.labels[0]);
                const int idx = K(x, y, el);

                if (not global_to_trace.contains(idx))
                {
                    global_to_trace[idx] = l;
                    P.push_back(idx);
                    ++l;
                }

                I(i, f) = global_to_trace[idx];
            }
        }

        ndof = global_to_trace.size();

        _proj.resize(ndof);
        auto proj = reshape(_proj.host_write(), ndof);
        for (int i = 0; i < ndof; ++i)
            proj(i) = P.at(i);
    }

    void TraceSpace2D::restrict(const double *__restrict__ x, double *__restrict__ y) const
    {
        const int n = ndof;
        auto proj = global_indices(MemorySpace::DEVICE);

        forall(n, [=] __device__(int i) -> void { y[i] = x[proj(i)]; });
    }

    void TraceSpace2D::prolong(const double *__restrict__ x, double *__restrict__ y) const
    {
        const int n = ndof;
        auto proj = global_indices(MemorySpace::DEVICE);

        forall(n, [=] __device__(int i) -> void { y[proj(i)] += x[i]; });
    }

    void TraceSpace2D::orth(double *x) const
    {
        auto proj = global_indices(MemorySpace::DEVICE);

        forall(ndof, [=] __device__(int i) -> void { x[proj(i)] = 0.0; });
    }

} // namespace cuddh
