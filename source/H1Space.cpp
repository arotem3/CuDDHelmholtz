#include "H1Space.hpp"

namespace cuddh
{
    H1Space::H1Space(const Mesh2D& mesh_, const Basis& basis_)
        : n_elem{mesh_.n_elem()},
          n_basis{basis_.size()},
          _mesh{mesh_},
          _basis{basis_},
          I(n_basis, n_basis, n_elem)
    {
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
            if (not mask.contains(i))
            {
                I[i] = l;
                ++l;
            }
        }

        for (auto [v1, v0] : mask)
        {
            I[v1] = I[v0];
        }
    }
} // namespace cuddh
