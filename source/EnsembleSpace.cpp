#include "EnsembleSpace.hpp"

template <typename Map, typename Key>
static bool contains(const Map & map, Key key)
{
    return map.find(key) != map.end();
}

namespace cuddh
{
    static auto compute_subspace_elements(int nel, int n_spaces, const int *element_labels)
    {
        std::vector<std::vector<int>> E(n_spaces); // elements

        for (int el = 0; el < nel; ++el)
        {
            const int p = element_labels[el];
            if (p < 0 || p >= n_spaces)
                cuddh_error("EnsembleSpace error: an element was illogically labeled.");
            E.at(p).push_back(el);
        }

        return E;
    }

    static ivec global_element_to_subspace_element(int nel, const std::vector<std::vector<int>> &E)
    {
        ivec el2s(nel); // maps global element index to subspace element index
        const int n_spaces = E.size();

        for (int p = 0; p < n_spaces; ++p)
        {
            auto &elems = E.at(p);
            const int n = elems.size();
            for (int i = 0; i < n; ++i)
            {
                el2s(elems.at(i)) = i;
            }
        }

        return el2s;
    }

    static int set_subdomain_num_elements(ivec_wrapper &h_s_elems, const std::vector<std::vector<int>> &E)
    {
        int mx = 0;
        int mn = 1;

        int n_spaces = E.size();
        for (int p = 0; p < n_spaces; ++p)
        {
            const int n = E.at(p).size();
            h_s_elems(p) = n;
            mx = std::max(mx, n);
            mn = std::min(mn, n);
        }

        if (mn < 1)
            cuddh_error("EnsembleSpace error: atleast one space is empty");

        return mx;
    }

    static void set_subdomain_elements(imat_wrapper &h_elems, const std::vector<std::vector<int>> &E)
    {
        const int n_spaces = E.size();

        std::fill(h_elems.begin(), h_elems.end(), -1);

        for (int p = 0; p < n_spaces; ++p)
        {
            auto &elems = E.at(p);
            const int n = elems.size();
            for (int i = 0; i < n; ++i)
            {
                h_elems(i, p) = elems.at(i);
            }
        }
    }

    static auto compute_subdomain_boundary_faces(const Mesh2D &mesh, int n_spaces, const int *element_labels)
    {
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

        return std::make_pair(std::move(F), std::move(shared_faces));
    }

    static int set_subdomain_num_boundary_faces(ivec_wrapper &h_s_faces, const std::vector<std::vector<std::pair<int,int>>> &F)
    {
        const int n_spaces = F.size();

        int mx = 0;
        for (int p = 0; p < n_spaces; ++p)
        {
            const int n = F.at(p).size();
            h_s_faces(p) = n;
            mx = std::max(mx, n);
        }

        return mx;
    }

    static void set_subdomain_face_indices(imat_wrapper &h_faces, const std::vector<std::vector<std::pair<int,int>>> &F)
    {
        const int n_spaces = F.size();
        std::fill(h_faces.begin(), h_faces.end(), -1);

        for (int p = 0; p < n_spaces; ++p)
        {
            auto &Fp = F.at(p);
            const int n = Fp.size();
            for (int i = 0; i < n; ++i)
            {
                auto [f, side] = Fp.at(i);
                h_faces(i, p) = f;
            }
        }
    }

    static auto set_subdomain_dof_indices(TensorWrapper<4,int> &h_sI, const H1Space &fem, const std::vector<std::vector<int>> &E)
    {
        const int n_spaces = E.size();
        const int n_basis = fem.basis().size();

        std::fill(h_sI.begin(), h_sI.end(), -1);

        std::vector<std::vector<int>> s2g(n_spaces); // subspace index to global index
        auto g_inds = fem.global_indices(MemorySpace::HOST); // global element indices

        for (int p = 0; p < n_spaces; ++p)
        {
            std::unordered_map<int, int> unique; // global index to subspace index
            auto &s2g_p = s2g.at(p);

            auto& subdomain_elements = E.at(p);
            const int n = subdomain_elements.size();

            int l = 0; // running index of subspace indices
            for (int el = 0; el < n; ++el)
            {
                const int g_el = subdomain_elements.at(el); // global element label
                for (int j = 0; j < n_basis; ++j)
                {
                    for (int i = 0; i < n_basis; ++i)
                    {
                        const int g_idx = g_inds(i, j, g_el); // global index
                        if (not contains(unique, g_idx))
                        {
                            unique[g_idx] = l;
                            s2g_p.push_back(g_idx);
                            ++l;
                        }

                        h_sI(i, j, el, p) = unique[g_idx];
                    }
                }
            }
        }

        return s2g;
    }

    static int set_subspace_sizes(ivec_wrapper &h_s_dof, const std::vector<std::vector<int>> &s2g)
    {
        const int n_spaces = s2g.size();

        int mx = 0;
        for (int p = 0; p < n_spaces; ++p)
        {
            const int n = s2g.at(p).size();
            h_s_dof(p) = n;
            mx = std::max(mx, n);
        }

        return mx;
    }

    static void set_global_indices(imat_wrapper &h_gI, const std::vector<std::vector<int>> &s2g)
    {
        const int n_spaces = s2g.size();

        std::fill(h_gI.begin(), h_gI.end(), -1);

        for (int p = 0; p < n_spaces; ++p)
        {
            auto& s_s2g = s2g.at(p);
            const int n = s_s2g.size();
            for (int i = 0; i < n; ++i)
            {
                h_gI(i, p) = s_s2g.at(i);
            }
        }
    }

    static auto compute_subdomain_fdof_indices(TensorWrapper<3, int> &h_fI, const H1Space &fem, const std::vector<std::vector<std::pair<int, int>>> &F, const ivec &el2s, const TensorWrapper<4, int> &h_sI)
    {
        const Mesh2D &mesh = fem.mesh();

        const int n_spaces = F.size();
        const int n_basis = fem.basis().size();

        std::fill(h_fI.begin(), h_fI.end(), -1);

        std::vector<std::vector<int>> f2s(n_spaces);
        for (int p = 0; p < n_spaces; ++p)
        {
            std::unordered_map<int, int> s_unique; // unique face indices
            auto& s_f2s = f2s.at(p);

            auto &Fp = F.at(p);
            const int nf = Fp.size();

            int l = 0;
            for (int f = 0; f < nf; ++f)
            {
                const auto [g_f, side] = Fp.at(f);

                const Edge * edge = mesh.edge(g_f);
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
        }

        return f2s;
    }

    static int set_subdomain_num_fdofs(ivec_wrapper &h_s_fdof, const std::vector<std::vector<int>> &f2s)
    {
        const int n_spaces = h_s_fdof.size();

        int mx = 0;
        for (int p = 0; p < n_spaces; ++p)
        {
            const int fdof = f2s.at(p).size();
            h_s_fdof(p) = fdof;
            mx = std::max(mx, fdof);
        }

        return mx;
    }

    static void map_dof_to_fdof(imat_wrapper &h_pI, const std::vector<std::vector<int>> &f2s)
    {
        const int n_spaces = f2s.size();

        std::fill(h_pI.begin(), h_pI.end(), -1);
        
        for (int p = 0; p < n_spaces; ++p)
        {
            auto& s_f2s = f2s.at(p);
            const int fdof = s_f2s.size();
            for (int i = 0; i < fdof; ++i)
            {
                h_pI(i, p) = s_f2s.at(i);
            }
        }
    }

    static int compute_shared_dof_map(HostDeviceArray<int> &cmap, const std::vector<std::array<int,4>> &shared_faces, const TensorWrapper<3,int> &h_fI, int n_spaces, int n_basis)
    {
        int n_shared = shared_faces.size(); // total number of faces shared between subdomains
        std::vector<std::array<int,4>> shared_dofs; // list of all pairs of shared DOFs identifying the respective subspaces
        std::unordered_map<int, std::unordered_set<int>> unique_shared; // maps pairs of subspaces to unique DOFs shared between them
        for (auto [S0, S1, f0, f1] : shared_faces)
        {
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

        int n_shared_dofs = shared_dofs.size();
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

        return n_shared_dofs;
    }

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
        auto E = compute_subspace_elements(nel, n_spaces, element_labels);
        auto el2s = global_element_to_subspace_element(nel, E);

        mx_elems = set_subdomain_num_elements(h_s_elems, E);
        
        // maps subspace element index to global element index
        elems.resize(mx_elems * n_spaces);
        auto h_elems = reshape(elems.host_write(), mx_elems, n_spaces);
        set_subdomain_elements(h_elems, E);

        // computes subdomain indices
        sI.resize(n_basis * n_basis * mx_elems * n_spaces);
        auto h_sI = reshape(sI.host_write(), n_basis, n_basis, mx_elems, n_spaces);
        auto s2g = set_subdomain_dof_indices(h_sI, fem, E);

        // determine the sizes of the subspaces
        mx_ndof = set_subspace_sizes(h_s_dof, s2g);

        // maps subspace indices to global indices
        gI.resize(mx_ndof * n_spaces);
        auto h_gI = reshape(gI.host_write(), mx_ndof, n_spaces);
        set_global_indices(h_gI, s2g);

        // determine faces in each subspace
        auto [F, shared_faces] = compute_subdomain_boundary_faces(mesh, n_spaces, element_labels);
        mx_faces = set_subdomain_num_boundary_faces(h_s_faces, F);

        // populate h_faces
        _faces.resize(mx_faces * n_spaces);
        auto h_faces = reshape(_faces.host_write(), mx_faces, n_spaces);
        set_subdomain_face_indices(h_faces, F);

        // fI maps the face local indices (with respect to subdomain face index)
        // to face space degrees of freedom
        fI.resize(n_basis * mx_faces * n_spaces);
        auto h_fI = reshape(fI.host_write(), n_basis, mx_faces, n_spaces);
        
        auto f2s = compute_subdomain_fdof_indices(h_fI, fem, F, el2s, h_sI);
        mx_fdof = set_subdomain_num_fdofs(h_s_fdof, f2s);

        // pI maps the subdomain face space degree of freedom to the subspace
        // degree of freedom
        pI.resize(mx_fdof * n_spaces);
        auto h_pI = reshape(pI.host_write(), mx_fdof, n_spaces);
        map_dof_to_fdof(h_pI, f2s);

        // determine the mapping between subdomain face spaces of the shared
        // degrees of freedom.
        n_shared_dofs = compute_shared_dof_map(cmap, shared_faces, h_fI, n_spaces, n_basis);
    }
} // namespace cuddh
