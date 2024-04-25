#include "EnsembleSpace.hpp"

template <typename Map, typename Key>
static bool contains(const Map & map, Key key)
{
    return map.find(key) != map.end();
}

namespace cuddh
{
    EnsembleSpace::EnsembleSpace(const H1Space& fem, int n_spaces_, const int * element_labels)
        : n_spaces{n_spaces_},
          n_basis{fem.basis().size()},
          s_dof(n_spaces),
          s_elems(n_spaces),
          s_faces(n_spaces),
          s_fdof(n_spaces)
    {
        auto& mesh = fem.mesh();
        const int nel = mesh.n_elem();

        auto h_s_elems = reshape(s_elems.host_write(), n_spaces);
        auto h_s_faces = reshape(s_faces.host_write(), n_spaces);
        auto h_s_dof = reshape(s_dof.host_write(), n_spaces);
        auto h_s_fdof = reshape(s_fdof.host_write(), n_spaces);

        // determine elements in each subspace
        std::vector<std::vector<int>> E(n_spaces); // elements
        ivec el2s(nel); // maps global element index to subspace element index
        for (int el = 0; el < nel; ++el)
        {
            const int p = element_labels[el];
            E.at(p).push_back(el);
            el2s(el) = E.at(p).size()-1;
        }

        int min_elems = 1;
        mx_elems = 0;
        for (int p = 0; p < n_spaces; ++p)
        {
            const int n = E.at(p).size();
            min_elems = std::min(min_elems, n);
            mx_elems = std::max(mx_elems, n);
            h_s_elems(p) = n;
        }

        if (min_elems < 1)
            cuddh_error("EnsembleSpace error: atleast one space is empty");
        
        // sI maps element local indices in subspace (with respect to subspace
        // element index) to subspace degree of freedom.
        sI.resize(n_basis * n_basis * mx_elems * n_spaces);
        auto h_sI = reshape(sI.host_write(), n_basis, n_basis, mx_elems, n_spaces);
        std::fill(h_sI.begin(), h_sI.end(), -1);
        
        // maps subspace element index to global element index
        elems.resize(mx_elems * n_spaces);
        auto h_elems = reshape(elems.host_write(), mx_elems, n_spaces);
        std::fill(h_elems.begin(), h_elems.end(), -1);
        
        for (int p = 0; p < n_spaces; ++p)
        {
            auto& _elems = E.at(p);
            const int n = h_s_elems(p);
            for (int i = 0; i < n; ++i)
            {
                h_elems(i, p) = _elems.at(i);
            }
        }

        // determine faces in each subspace
        std::vector<std::vector<std::pair<int,int>>> F(n_spaces); // faces in each subspace
        std::vector<std::array<int,4>> shared_faces; // {subdomain0, subdomain1, subdomain face index0, ..face..1}
        const int g_faces = mesh.n_edges(); // global number of faces
        for (int e = 0; e < g_faces; ++e)
        {
            // loop over faces and check if an edge is on the boundary of a
            // subdomain. Boundary faces are automatically on the boundary, and
            // interior faces are on the boundary only if the element[0] != element[1].

            const Edge * edge = mesh.edge(e);
            
            const int el0 = edge->elements[0];
            const int S0 = element_labels[el0];

            if (edge->type == FaceType::BOUNDARY)
            {
                F.at(S0).push_back({e, 0});
            }
            else
            {
                const int el1 = edge->elements[1];
                const int S1 = element_labels[el1];

                if (S0 != S1)
                {
                    F.at(S0).push_back({e, 0});
                    F.at(S1).push_back({e, 1});

                    const int l0 = F.at(S0).size() - 1; // the index of face e in the subdomain face space
                    const int l1 = F.at(S1).size() - 1;
                    shared_faces.push_back({S0, S1, l0, l1});
                }
            }
        }

        mx_faces = 0;
        for (int p = 0; p < n_spaces; ++p)
        {
            const int n = F.at(p).size();
            mx_faces = std::max(mx_faces, n);
            h_s_faces(p) = n;
        }

        // fI maps the face local indices (with respect to subdomain face index)
        // to face space degrees of freedom
        fI.resize(n_basis * mx_faces * n_spaces);
        auto h_fI = reshape(fI.host_write(), n_basis, mx_faces, n_spaces);
        std::fill(h_fI.begin(), h_fI.end(), -1);

        _faces.resize(mx_faces * n_spaces);
        auto h_faces = reshape(_faces.host_write(), mx_faces, n_spaces);
        std::fill(h_faces.begin(), h_faces.end(), -1);

        imat face_side(mx_faces, n_spaces);
        std::fill(face_side.begin(), face_side.end(), -1);

        for (int p = 0; p < n_spaces; ++p)
        {
            const int n = h_s_faces(p);
            for (int i = 0; i < n; ++i)
            {
                auto [f, side] = F.at(p).at(i);
                h_faces(i, p) = f;
                face_side(i, p) = side;
            }
        }

        // determine the mapping from subspace indices to global indices.
        std::vector<std::vector<int>> s2g(n_spaces); // subspace index to global index
        auto g_inds = fem.global_indices(MemorySpace::HOST); // global element indices
        mx_ndof = 0;
        for (int p = 0; p < n_spaces; ++p)
        {
            std::unordered_map<int, int> s_unique;
            auto& s_s2g = s2g.at(p);

            int l = 0; // running index of subspace indices
            const int n = h_s_elems(p);
            for (int el = 0; el < n; ++el)
            {
                const int g_el = h_elems(el, p); // global element label
                for (int j = 0; j < n_basis; ++j)
                {
                    for (int i = 0; i < n_basis; ++i)
                    {
                        const int g_idx = g_inds(i, j, g_el); // global index
                        if (not contains(s_unique, g_idx))
                        {
                            s_unique[g_idx] = l;
                            s_s2g.push_back(g_idx);
                            ++l;
                        }

                        h_sI(i, j, el, p) = s_unique[g_idx];
                    }
                }
            }

            const int ndof = s_unique.size(); // number of DOFs in subspace
            mx_ndof = std::max(mx_ndof, ndof);
            h_s_dof(p) = ndof;
        }
        
        // maps subspace indices to global indices
        gI.resize(mx_ndof * n_spaces);
        auto h_gI = reshape(gI.host_write(), mx_ndof, n_spaces);
        std::fill(h_gI.begin(), h_gI.end(), -1);

        for (int p = 0; p < n_spaces; ++p)
        {
            auto& s_s2g = s2g.at(p);
            const int n = h_s_dof(p);
            for (int i = 0; i < n; ++i)
            {
                h_gI(i, p) = s_s2g.at(i);
            }
        }

        // determine the mapping from subdomain face space indices to subspace indices
        std::vector<std::vector<int>> f2s(n_spaces);
        mx_fdof = 0;
        for (int p = 0; p < n_spaces; ++p)
        {
            std::unordered_map<int, int> s_unique; // unique face indices
            auto& s_f2s = f2s.at(p);

            int l = 0;
            const int nf = h_s_faces(p);
            for (int f = 0; f < nf; ++f)
            {
                const Edge * edge = mesh.edge(h_faces(f, p));
                const int side = face_side(f, p);
                const int g_el = edge->elements[side];
                const int s = edge->sides[side];
                const bool reversed = (side == 1 && edge->delta < 0);

                for (int i = 0; i < n_basis; ++i)
                {
                    // map face index to element index
                    const int j = (reversed) ? (n_basis-1-i) : i;
                    const int m = (s == 0 || s == 2) ? j : (s == 1) ? (n_basis-1) : 0;
                    const int n = (s == 1 || s == 3) ? j : (s == 2) ? (n_basis-1) : 0;
                    const int el = el2s(g_el);
                    
                    const int idx = h_sI(m, n, el, p);

                    if (not contains(s_unique, idx))
                    {
                        s_unique[idx] = l;
                        s_f2s.push_back(idx);
                        ++l;
                    }

                    h_fI(i, f, p) = s_unique.at(idx);
                }
            }

            const int fdof = s_unique.size();
            h_s_fdof(p) = fdof;
            mx_fdof = std::max(mx_fdof, fdof);
        }

        // pI maps the subdomain face space degree of freedom to the subspace
        // degree of freedom
        pI.resize(mx_fdof * n_spaces);
        auto h_pI = reshape(pI.host_write(), mx_fdof, n_spaces);
        std::fill(h_pI.begin(), h_pI.end(), -1);
        
        for (int p = 0; p < n_spaces; ++p)
        {
            auto& s_f2s = f2s.at(p);
            const int fdof = h_s_fdof(p);
            for (int i = 0; i < fdof; ++i)
            {
                h_pI(i, p) = s_f2s.at(i);
            }
        }

        // determine the mapping between subdomain face spaces of the shared
        // degrees of freedom.
        int n_shared = shared_faces.size(); // total number of faces shared between subdomains
        std::vector<std::array<int,4>> shared_dofs(n_spaces); // list of all pairs of shared DOFs identifying the respective subspaces
        std::unordered_map<int, std::unordered_set<int>> unique_shared; // maps pairs of subspaces to unique DOFs shared between them
        for (int f = 0; f < n_shared; ++f)
        {
            auto [S0, S1, f0, f1] = shared_faces.at(f);
            const int key = (S0 < S1) ? (S0 + n_spaces * S1) : (S1 + n_spaces * S0); // key is same for (S0, S1) and (S1, S0) symmetric pairs

            auto& unq = unique_shared[key]; // unique face dofs
            for (int i = 0; i < n_basis; ++i)
            {
                const int j0 = h_fI(i, f0, S0);
                const int j1 = h_fI(i, f1, S1);

                const int lkey = (S0 < S1) ? j0 : j1; // key is same for symmetric pairs
                if (not contains(unq, lkey))
                {
                    shared_dofs.push_back({S0, S1, j0, j1});
                    unq.insert(lkey);
                }
            }
        }

        n_shared_dofs = shared_dofs.size();
        cmap.resize(4 * n_shared_dofs);
        auto h_cmap = reshape(cmap.host_write(), 4, n_shared_dofs);
        for (int i = 0; i < n_shared_dofs; ++i)
        {
            auto [S0, S1, j0, j1] = shared_dofs.at(i);
            h_cmap(0, i) = S0;
            h_cmap(1, i) = S1;
            h_cmap(2, i) = j0;
            h_cmap(3, i) = j1;
        }
    }
} // namespace cuddh
