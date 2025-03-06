#include "DD2D/EnsembleSpace.hpp"

using namespace cuddh;

class EnsembleSpaceBuilder
{
public:
    EnsembleSpaceBuilder(const H1Space2D &fem, int n_spaces, const int *element_labels);

    int set_subdomain_num_elements(ivec_wrapper &h_s_elems) const;
    void set_subdomain_elements(imat_wrapper &h_elems) const;
    int set_subdomain_num_boundary_faces(ivec_wrapper &h_s_faces) const;
    void set_subdomain_face_indices(imat_wrapper &h_faces) const;
    int compute_shared_dof_map(HostDeviceArray<int> &cmap, const TensorWrapper<3, int> &h_fI) const;
    void compute_fdof_indices(TensorWrapper<3, int> &h_fI, const TensorWrapper<4, int> &h_sI) const;
    int set_subdomain_num_dofs(ivec_wrapper &h_s_dof) const;
    int set_subdomain_num_fdofs(ivec_wrapper &h_s_fdof) const;
    void compute_dof_indices(TensorWrapper<4, int> &h_sI) const;
    void compute_global_indices(imat_wrapper &h_gI) const;

private:
    const H1Space2D &fem;
    std::vector<std::vector<int>> E;                 // elements
    std::vector<std::vector<std::pair<int, int>>> F; // faces
    std::vector<std::array<int, 4>> shared_faces;    // {subdomain0, subdomain1, subdomain face index0, ..face..1}
    std::vector<std::vector<int>> s2g;               // subspace index to global index
    std::vector<std::vector<int>> f2s;               // face index to subspace index
};

EnsembleSpace::EnsembleSpace(const H1Space2D &fem, int n_spaces_, const int *element_labels)
    : n_spaces{n_spaces_},
      n_basis{fem.basis().size()},
      s_dof(n_spaces),
      s_elems(n_spaces),
      s_faces(n_spaces),
      s_fdof(n_spaces)
{
    auto &mesh = fem.mesh();
    const int nel = mesh.n_elem();

    auto h_s_elems = reshape(s_elems.host_write(), n_spaces);
    auto h_s_faces = reshape(s_faces.host_write(), n_spaces);
    auto h_s_dof = reshape(s_dof.host_write(), n_spaces);
    auto h_s_fdof = reshape(s_fdof.host_write(), n_spaces);

    // determine elements in each subspace
    EnsembleSpaceBuilder ESbuilder(fem, n_spaces, element_labels);

    mx_elems = ESbuilder.set_subdomain_num_elements(h_s_elems);

    // maps subspace element index to global element index
    elems.resize(mx_elems * n_spaces);
    auto h_elems = reshape(elems.host_write(), mx_elems, n_spaces);
    ESbuilder.set_subdomain_elements(h_elems);

    // determine faces in each subspace
    mx_faces = ESbuilder.set_subdomain_num_boundary_faces(h_s_faces);

    // populate h_faces
    _faces.resize(mx_faces * n_spaces);
    auto h_faces = reshape(_faces.host_write(), mx_faces, n_spaces);
    ESbuilder.set_subdomain_face_indices(h_faces);

    // determine the mapping between global and subspace indices
    sI.resize(n_basis * n_basis * mx_elems * n_spaces);
    auto h_sI = reshape(sI.host_write(), n_basis, n_basis, mx_elems, n_spaces);
    ESbuilder.compute_dof_indices(h_sI);

    mx_ndof = ESbuilder.set_subdomain_num_dofs(h_s_dof);
    mx_fdof = ESbuilder.set_subdomain_num_fdofs(h_s_fdof);

    gI.resize(mx_ndof * n_spaces);
    auto h_gI = reshape(gI.host_write(), mx_ndof, n_spaces);
    ESbuilder.compute_global_indices(h_gI);

    fI.resize(n_basis * mx_faces * n_spaces);
    auto h_fI = reshape(fI.host_write(), n_basis, mx_faces, n_spaces);
    ESbuilder.compute_fdof_indices(h_fI, h_sI);

    // determine the mapping between subdomain face spaces of the shared
    // degrees of freedom.
    n_shared_dofs = ESbuilder.compute_shared_dof_map(cmap, h_fI);
}

EnsembleSpace cuddh::partition_uniform_rect(const H1Space2D &fem, int nx, int ny, int max_dof_1d)
{
    const int n_basis = fem.basis().size();
    const int elems_per_domain_x = max_dof_1d / n_basis;

    if (nx % elems_per_domain_x != 0 || ny % elems_per_domain_x != 0)
        cuddh_error("Only nx x ny meshes with nx and ny multiples of 32 / n_basis allowed.");

    const int num_domains_x = nx / elems_per_domain_x;
    const int num_domains_y = ny / elems_per_domain_x;

    int n_domains = num_domains_x * num_domains_y;

    imat element_labels(nx, ny);
    std::fill(element_labels.begin(), element_labels.end(), -1);

    for (int j = 0; j < ny; ++j)
    {
        for (int i = 0; i < nx; ++i)
        {
            int label_x = i / elems_per_domain_x;
            int label_y = j / elems_per_domain_x;
            element_labels(i, j) = label_x + num_domains_x * label_y;
        }
    }

    return EnsembleSpace(fem, n_domains, element_labels);
}

template <typename Map, typename Key>
static bool contains(const Map &map, Key key)
{
    return map.find(key) != map.end();
}

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

static auto compute_subdomain_boundary_faces(const Mesh2D &mesh, int n_spaces, const int *element_labels)
{
    std::vector<std::vector<std::pair<int, int>>> F(n_spaces); // faces in each subspace
    std::vector<std::array<int, 4>> shared_faces;              // {subdomain0, subdomain1, subdomain face index0, ..face..1}
    const int g_faces = mesh.n_edges();                        // global number of faces
    for (int e = 0; e < g_faces; ++e)
    {
        // loop over faces and check if an edge is on the boundary of a
        // subdomain. Boundary faces are automatically on the boundary, and
        // interior faces are on the boundary only if the element[0] != element[1].

        const Edge *edge = mesh.edge(e);

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

static void natural_ordering(std::vector<int> &dof_indices, std::vector<int> &fdof_indices)
{
    const int ndof = dof_indices.size();
    const int fdof = fdof_indices.size();

    std::unordered_set<int> pp;
    std::vector<int> perm(ndof);

    int l = 0;
    for (; l < fdof; ++l)
    {
        int j = fdof_indices.at(l);
        pp.insert(j);
        perm.at(l) = j;
    }

    for (int i = 0; i < ndof; ++i)
    {
        if (contains(pp, i))
            continue;

        perm.at(l) = i;
        ++l;
    }

    std::vector<int> I = dof_indices;
    for (int i = 0; i < ndof; ++i)
    {
        dof_indices.at(i) = I.at(perm.at(i));
    }
}

EnsembleSpaceBuilder::EnsembleSpaceBuilder(const H1Space2D &fem, int n_spaces, const int *element_labels)
    : fem{fem}
{
    E = compute_subspace_elements(fem.mesh().n_elem(), n_spaces, element_labels);
    std::tie(F, shared_faces) = compute_subdomain_boundary_faces(fem.mesh(), n_spaces, element_labels);

    const Mesh2D &mesh = fem.mesh();

    const int n_basis = fem.basis().size();
    s2g.resize(n_spaces);
    f2s.resize(n_spaces);

    auto g_inds = fem.global_indices(MemorySpace::HOST); // global element indices

    for (int p = 0; p < n_spaces; ++p)
    {
        std::unordered_map<int, int> unique; // global index to subspace index
        auto &dof_indices = s2g.at(p);

        auto &subdomain_elements = E.at(p);
        const int n_elem = subdomain_elements.size();

        int l = 0; // running index of subspace indices
        for (int el = 0; el < n_elem; ++el)
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
                        dof_indices.push_back(g_idx);
                        ++l;
                    }
                }
            }
        }

        std::unordered_map<int, int> funique; // unique face indices
        auto &fdof_indices = f2s.at(p);

        auto &subdomain_faces = F.at(p);
        const int nf = subdomain_faces.size();

        l = 0;
        for (int f = 0; f < nf; ++f)
        {
            const auto [g_f, side] = subdomain_faces.at(f);

            const Edge *edge = mesh.edge(g_f);
            const int g_el = edge->elements[side];
            const int s = edge->sides[side];
            const bool reversed = (side == 1 && edge->delta < 0);

            for (int i = 0; i < n_basis; ++i)
            {
                // map face index to element index
                const int j = (reversed) ? (n_basis - 1 - i) : i;
                const int m = (s == 0 || s == 2) ? j : (s == 1) ? (n_basis - 1)
                                                                : 0;
                const int n = (s == 1 || s == 3) ? j : (s == 2) ? (n_basis - 1)
                                                                : 0;

                const int idx = unique.at(g_inds(m, n, g_el));

                if (not contains(funique, idx))
                {
                    funique[idx] = l;
                    fdof_indices.push_back(idx);
                    ++l;
                }
            }
        }

        natural_ordering(dof_indices, fdof_indices);
    }
}

int EnsembleSpaceBuilder::set_subdomain_num_elements(ivec_wrapper &h_s_elems) const
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

void EnsembleSpaceBuilder::set_subdomain_elements(imat_wrapper &h_elems) const
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

int EnsembleSpaceBuilder::set_subdomain_num_boundary_faces(ivec_wrapper &h_s_faces) const
{
    const int n_spaces = E.size();

    int mx = 0;
    for (int p = 0; p < n_spaces; ++p)
    {
        const int n = F.at(p).size();
        h_s_faces(p) = n;
        mx = std::max(mx, n);
    }

    return mx;
}

void EnsembleSpaceBuilder::set_subdomain_face_indices(imat_wrapper &h_faces) const
{
    const int n_spaces = E.size();
    std::fill(h_faces.begin(), h_faces.end(), -1);

    for (int p = 0; p < n_spaces; ++p)
    {
        auto &sf = F.at(p);
        const int nf = sf.size();
        for (int i = 0; i < nf; ++i)
        {
            auto [f, side] = sf.at(i);
            h_faces(i, p) = f;
        }
    }
}

int EnsembleSpaceBuilder::compute_shared_dof_map(HostDeviceArray<int> &cmap, const TensorWrapper<3, int> &h_fI) const
{
    const int n_shared = shared_faces.size(); // total number of faces shared between subdomains
    const int n_basis = fem.basis().size();
    const int n_spaces = E.size();

    std::vector<std::array<int, 4>> shared_dofs;                    // list of all pairs of shared DOFs identifying the respective subspaces
    std::unordered_map<int, std::unordered_set<int>> unique_shared; // maps pairs of subspaces to unique DOFs shared between them

    for (auto [S0, S1, f0, f1] : shared_faces)
    {
        const int key = (S0 < S1) ? (S0 + n_spaces * S1) : (S1 + n_spaces * S0); // key is same for (S0, S1) and (S1, S0) symmetric pairs

        auto &unq = unique_shared[key]; // unique face dofs
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

void EnsembleSpaceBuilder::compute_fdof_indices(TensorWrapper<3, int> &h_fI, const TensorWrapper<4, int> &h_sI) const
{
    const Mesh2D &mesh = fem.mesh();
    const Basis &basis = fem.basis();

    const int n_basis = basis.size();

    auto el2s = global_element_to_subspace_element(mesh.n_elem(), E);

    const int n_spaces = E.size();

    for (int p = 0; p < n_spaces; ++p)
    {
        auto &sf = F.at(p);
        const int nf = sf.size();

        for (int f = 0; f < nf; ++f)
        {
            const auto [g_f, side] = sf.at(f);

            const Edge *edge = mesh.edge(g_f);
            const int g_el = edge->elements[side];
            const int s = edge->sides[side];
            const bool reversed = (side == 1 && edge->delta < 0);

            const int el = el2s(g_el);

            for (int i = 0; i < n_basis; ++i)
            {
                // map face index to element index
                const int j = (reversed) ? (n_basis - 1 - i) : i;
                const int m = (s == 0 || s == 2) ? j : (s == 1) ? (n_basis - 1)
                                                                : 0;
                const int n = (s == 1 || s == 3) ? j : (s == 2) ? (n_basis - 1)
                                                                : 0;

                const int idx = h_sI(m, n, el, p);

                h_fI(i, f, p) = idx;
            }
        }
    }
}

int EnsembleSpaceBuilder::set_subdomain_num_dofs(ivec_wrapper &h_s_dof) const
{
    const int n_spaces = E.size();

    int mx = 0;
    for (int p = 0; p < n_spaces; ++p)
    {
        const int n = s2g.at(p).size();
        h_s_dof(p) = n;
        mx = std::max(mx, n);
    }

    return mx;
}

int EnsembleSpaceBuilder::set_subdomain_num_fdofs(ivec_wrapper &h_s_fdof) const
{
    const int n_spaces = E.size();

    int mx = 0;
    for (int p = 0; p < n_spaces; ++p)
    {
        const int n = f2s.at(p).size();
        h_s_fdof(p) = n;
        mx = std::max(mx, n);
    }

    return mx;
}

void EnsembleSpaceBuilder::compute_dof_indices(TensorWrapper<4, int> &h_sI) const
{
    const int n_spaces = E.size();
    const int n_basis = fem.basis().size();

    for (int p = 0; p < n_spaces; ++p)
    {
        std::unordered_map<int, int> inv_indices; // global index to subspace index

        auto &dof_indices = s2g.at(p);
        auto &subdomain_elements = E.at(p);

        const int n_elem = subdomain_elements.size();
        const int ndof = dof_indices.size();

        auto g_inds = fem.global_indices(MemorySpace::HOST); // global element indices

        for (int i = 0; i < ndof; ++i)
            inv_indices[dof_indices.at(i)] = i;

        for (int el = 0; el < n_elem; ++el)
        {
            const int g_el = subdomain_elements.at(el);
            for (int j = 0; j < n_basis; ++j)
            {
                for (int i = 0; i < n_basis; ++i)
                {
                    const int g_idx = g_inds(i, j, g_el);
                    h_sI(i, j, el, p) = inv_indices.at(g_idx);
                }
            }
        }
    }
}

void EnsembleSpaceBuilder::compute_global_indices(imat_wrapper &h_gI) const
{
    const int n_spaces = E.size();

    for (int p = 0; p < n_spaces; ++p)
    {
        auto &s_s2g = s2g.at(p);
        const int n = s_s2g.size();
        for (int i = 0; i < n; ++i)
        {
            h_gI(i, p) = s_s2g.at(i);
        }
    }
}
