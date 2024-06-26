#include "H1Space.hpp"

template <typename Map, typename Key>
static bool contains(const Map & map, Key key)
{
    return map.find(key) != map.end();
}

namespace cuddh
{
    H1Space::H1Space(const Mesh2D& mesh_, const Basis& basis_)
        : n_elem{mesh_.n_elem()},
          n_basis{basis_.size()},
          _mesh{mesh_},
          _basis{basis_},
          _I(n_basis * n_basis * n_elem)
    {
        icube_wrapper I(_I.host_write(), n_basis, n_basis, n_elem);

        std::unordered_map<int, int> mask;
        
        const int n_edges = _mesh.n_edges(FaceType::INTERIOR);
        const int n_nodes = _mesh.n_nodes();

        // map edge index to volume index
        const int nc = n_basis;
        auto E2V = [nc](int i, int f, int el) -> int
        {
            const int m = (f == 0 || f == 2) ? i : (f == 1) ? (nc-1) : 0;
            const int n = (f == 1 || f == 3) ? i : (f == 2) ? (nc-1) : 0;

            return m + nc * (n + nc * el);
        };

        // map node to volume index
        auto N2V = [nc](int c, int el) -> int
        {
            const int m = (c == 0 || c == 3) ? 0 : (nc-1);
            const int n = (c == 0 || c == 1) ? 0 : (nc-1);
            
            return m + nc * (n + nc * el);
        };

        // iterate over interior edges to indentify duplicates DOFs
        if (n_basis > 2)
        {
            for (int e = 0; e < n_edges; ++e)
            {
                auto edge = _mesh.edge(e, FaceType::INTERIOR);

                const int el0 = edge->elements[0];
                const int s0 = edge->sides[0];

                const int el1 = edge->elements[1];
                const int s1 = edge->sides[1];

                const bool reversed = edge->delta < 0;

                for (int i = 1; i < n_basis-1; ++i)
                {
                    const int j = (reversed) ? (n_basis-1-i) : i;

                    const int v0 = E2V(i, s0, el0);
                    const int v1 = E2V(j, s1, el1);
                    mask[v1] = v0;
                }
            }
        }

        // iterate over nodes to identify duplicate DOFs
        for (int k = 0; k < n_nodes; ++k)
        {
            auto& node = _mesh.node(k);
            
            const int nel = node.connected_elements.size();
            const int el0 = node.connected_elements.at(0).id;
            const int c0 = node.connected_elements.at(0).i;

            const int v0 = N2V(c0, el0);

            for (int i = 1; i < nel; ++i)
            {
                const int el = node.connected_elements.at(i).id;
                const int c = node.connected_elements.at(i).i;

                const int vi = N2V(c, el);
                mask[vi] = v0;
            }
        }

        const int N = n_elem * n_basis * n_basis;
        ndof = N - mask.size();
        int l = 0;
        for (int i = 0; i < N; ++i)
        {
            if (not contains(mask, i))
            {
                I[i] = l;
                ++l;
            }
        }

        for (auto [v1, v0] : mask)
        {
            I[v1] = I[v0];
        }

        _xy.resize(2 * ndof);
        auto xy = reshape(_xy.host_write(), 2, ndof);

        for (int el = 0; el < n_elem; ++el)
        {
            const Element * elem = _mesh.element(el);
            for (int j = 0; j < n_basis; ++j)
            {
                for (int i = 0; i < n_basis; ++i)
                {
                    const double xi[2] = {_basis.quadrature().x(i), _basis.quadrature().x(j)};
                    double x[2];
                    elem->physical_coordinates(xi, x);
                    const int idx = I(i, j, el);
                    xy(0, idx) = x[0];
                    xy(1, idx) = x[1];
                }
            }
        }
    }

    FaceSpace::FaceSpace(const H1Space& fem_, int nf, const int * faces_)
        : fem{fem_},
          _n_faces{nf},
          n_basis{fem.basis().size()},
          _I(n_basis * nf),
          _faces(nf)
    {
        auto F = reshape(_faces.host_write(), nf);
        auto I = reshape(_I.host_write(), n_basis, nf);

        for (int i = 0; i < nf; ++i)
            F(i) = faces_[i];

        const Mesh2D& mesh = fem.mesh();
        const int n_elem = mesh.n_elem();
        auto K = reshape(fem.global_indices(MemorySpace::HOST), n_basis, n_basis, n_elem);

        std::unordered_map<int, int> mask; // unique mapping from global DOFs to restricted DOFs
        std::vector<int> P;

        // map edge index to volume index
        const int nc = n_basis;
        auto E2V = [nc](int i, int f, int el) -> int
        {
            const int m = (f == 0 || f == 2) ? i : (f == 1) ? (nc-1) : 0;
            const int n = (f == 1 || f == 3) ? i : (f == 2) ? (nc-1) : 0;

            return m + nc * (n + nc * el);
        };
        
        int l = 0;
        for (int f = 0; f < nf; ++f)
        {
            const Edge * edge = mesh.edge(F(f));
            const int el = edge->elements[0];
            const int s = edge->sides[0];

            for (int i = 0; i < n_basis; ++i)
            {
                const int idx = K[E2V(i, s, el)];

                if (not contains(mask, idx))
                {
                    mask[idx] = l;
                    P.push_back(idx);
                    ++l;
                }

                I(i, f) = mask[idx];
            }
        }

        ndof = mask.size();

        _proj.resize(ndof);
        auto proj = reshape(_proj.host_write(), ndof);
        for (int i = 0; i < ndof; ++i)
            proj(i) = P.at(i);
    }

    void FaceSpace::restrict(const double * __restrict__ x, double * __restrict__ y) const
    {
        const int n = ndof;
        auto proj = global_indices(MemorySpace::DEVICE);

        forall(n, [=] __device__ (int i) -> void
        {
            y[i] = x[proj(i)];
        });
    }

    void FaceSpace::prolong(const double * __restrict__ x, double * __restrict__ y) const
    {
        const int n = ndof;
        auto proj = global_indices(MemorySpace::DEVICE);

        forall(n, [=] __device__ (int i) -> void
        {
            y[proj(i)] += x[i];
        });
    }

    void FaceSpace::orth(double * x) const
    {
        auto proj = global_indices(MemorySpace::DEVICE);

        forall(ndof, [=] __device__ (int i) -> void
        {
            x[proj(i)] = 0.0;
        });
    }

    const Mesh2D::EdgeMetricCollection& FaceSpace::metrics(const QuadratureRule& quad) const
    {
        auto key = quad.name();
        if (not contains(_metrics, key))
        {
            _metrics.insert({key, Mesh2D::EdgeMetricCollection(fem.mesh(), _n_faces, _faces.host_read(), quad)});
        }
        return _metrics.at(key);
    }
} // namespace cuddh
