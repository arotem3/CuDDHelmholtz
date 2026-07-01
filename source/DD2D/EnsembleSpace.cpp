#include "DD2D/EnsembleSpace.hpp"

#include "SparseMatrix.hpp"

using namespace cuddh;

namespace
{
    class EnsembleSpaceBuilder
    {
    public:
        EnsembleSpaceBuilder(const H1Space2D &fem, int n_spaces, const int *element_labels);

        int set_subdomain_num_elements(ivec_wrapper &h_s_elems) const;
        void set_subdomain_elements(imat_wrapper &h_elems) const;
        int set_subdomain_num_boundary_faces(ivec_wrapper &h_s_faces) const;
        void set_subdomain_face_indices(imat_wrapper &h_faces) const;
        void set_subdomain_face_sides(imat_wrapper &h_f_sides) const;
        int n_shared_faces() const { return shared_faces.size(); }
        int set_shared_faces(imat_wrapper &h_shared_faces) const;
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
} // namespace

EnsembleSpace::EnsembleSpace(const H1Space2D &fem, int n_spaces_, const int *element_labels)
    : fem{fem},
      n_spaces{n_spaces_},
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

    _element_labels.resize(nel);
    auto h_labels = reshape(_element_labels.host_write(), nel);
    for (int el = 0; el < nel; ++el)
        h_labels(el) = element_labels[el];

    // determine faces in each subspace
    mx_faces = ESbuilder.set_subdomain_num_boundary_faces(h_s_faces);

    // populate h_faces
    _faces.resize(mx_faces * n_spaces);
    auto h_faces = reshape(_faces.host_write(), mx_faces, n_spaces);
    ESbuilder.set_subdomain_face_indices(h_faces);

    // store which side (0 or 1) of each face belongs to each subdomain
    f_sides.resize(mx_faces * n_spaces);
    auto h_f_sides = reshape(f_sides.host_write(), mx_faces, n_spaces);
    ESbuilder.set_subdomain_face_sides(h_f_sides);

    n_shared_faces = ESbuilder.n_shared_faces();
    _shared_faces.resize(4 * n_shared_faces);
    auto h_shared_faces = reshape(_shared_faces.host_write(), 4, n_shared_faces);
    ESbuilder.set_shared_faces(h_shared_faces);

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
}

static constexpr int2 get_block_dims(int2 block_dims, int n_basis)
{
    const int edof = n_basis * n_basis;

    auto [bx, by] = block_dims;

    if (bx > 0 && by > 0)
        return {bx, by};

    if (bx > 0)
        by = std::max(1, CUDDH_DD2D_MX_DOF / (edof * bx));
    else if (by > 0)
        bx = std::max(1, CUDDH_DD2D_MX_DOF / (edof * by));
    else // Heuristic selection
    {
        by = std::sqrt((double)CUDDH_DD2D_MX_DOF / edof);
        bx = std::max(1, CUDDH_DD2D_MX_DOF / (edof * by));
    }

    return {bx, by};
}

EnsembleSpace cuddh::partition_uniform_rect(const H1Space2D &fem, int2 mesh_dims, int2 block_dims)
{
    const int n_basis = fem.basis().size();
    const auto [nx, ny] = mesh_dims;
    const auto [bx, by] = get_block_dims(block_dims, n_basis);

    const int dx = (nx + bx - 1) / bx;
    const int dy = (ny + by - 1) / by;

    int nd = dx * dy;

    imat element_labels(nx, ny);
    std::fill(element_labels.begin(), element_labels.end(), -1);

    for (int y = 0; y < ny; ++y)
    {
        for (int x = 0; x < nx; ++x)
        {
            int label_x = x / bx;
            int label_y = y / by;

            element_labels(x, y) = label_x + dx * label_y;
        }
    }

    return EnsembleSpace(fem, nd, element_labels);
}

static auto compute_subspace_elements(int nel, int n_spaces, const int *element_labels)
{
    std::vector<std::vector<int>> E(n_spaces); // elements

    for (int el = 0; el < nel; ++el)
    {
        const int p = element_labels[el];
        cuddh_assert(0 <= p && p < n_spaces, printf("EnsembleSpace error: an element was illogically labeled."));
        E.at(p).push_back(el);
    }

    return E;
}

static auto compute_subdomain_boundary_faces(const Mesh2D &mesh, int n_spaces, const int *element_labels)
{
    std::vector<std::vector<std::pair<int, int>>> F(n_spaces); // faces in each subspace
    std::vector<std::array<int, 4>> shared_faces; // {subdomain0, subdomain1, subdomain face index0, ..face..1}
    const int g_faces = mesh.n_edges();           // global number of faces

    for (int face_index = 0; face_index < g_faces; ++face_index)
    {
        // loop over faces and check if an edge is on the boundary of a
        // subdomain. Boundary faces are automatically on the boundary, and
        // interior faces are on the boundary only if the element[0] != element[1].

        const EdgeConnectivity edge = mesh.edge_connectivity(face_index);

        const int el0 = edge.elements[0];
        const int domain0 = element_labels[el0];

        if (edge.elements[1] == -1) // boundary edge
        {
            F.at(domain0).push_back({face_index, 0});
        }
        else
        {
            const int el1 = edge.elements[1];
            const int domain1 = element_labels[el1];

            if (domain0 != domain1)
            {
                F.at(domain0).push_back({face_index, 0});
                F.at(domain1).push_back({face_index, 1});

                const int local_face_index0 = F.at(domain0).size() - 1;
                const int local_face_index1 = F.at(domain1).size() - 1;
                shared_faces.push_back({domain0, domain1, local_face_index0, local_face_index1});
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
        if (pp.contains(i))
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

::EnsembleSpaceBuilder::EnsembleSpaceBuilder(const H1Space2D &fem, int n_spaces, const int *element_labels) : fem{fem}
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
                    if (not unique.contains(g_idx))
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

            const EdgeConnectivity edge = mesh.edge_connectivity(g_f);
            const int g_el = edge.elements[side];

            for (int i = 0; i < n_basis; ++i)
            {
                // map face index to element index
                const int j = (side == 1) ? permute_edge_index(n_basis, i, edge.permutation) : i;
                const auto [m, n] = edge2vol(n_basis, j, edge.labels[side]);

                const int idx = unique.at(g_inds(m, n, g_el));

                if (not funique.contains(idx))
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

int ::EnsembleSpaceBuilder::set_subdomain_num_elements(ivec_wrapper &h_s_elems) const
{
    int mx = 0;

    int n_spaces = E.size();
    for (int p = 0; p < n_spaces; ++p)
    {
        const int n = E.at(p).size();
        h_s_elems(p) = n;
        mx = std::max(mx, n);

        cuddh_verify(n >= 1, printf("EnsembleSpace error: Subspace %d is empty.\n", p));
    }

    return mx;
}

void ::EnsembleSpaceBuilder::set_subdomain_elements(imat_wrapper &h_elems) const
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

int ::EnsembleSpaceBuilder::set_subdomain_num_boundary_faces(ivec_wrapper &h_s_faces) const
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

void ::EnsembleSpaceBuilder::set_subdomain_face_indices(imat_wrapper &h_faces) const
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

void ::EnsembleSpaceBuilder::set_subdomain_face_sides(imat_wrapper &h_f_sides) const
{
    const int n_spaces = E.size();
    std::fill(h_f_sides.begin(), h_f_sides.end(), -1);

    for (int p = 0; p < n_spaces; ++p)
    {
        auto &sf = F.at(p);
        const int nf = sf.size();
        for (int i = 0; i < nf; ++i)
        {
            auto [f, side] = sf.at(i);
            h_f_sides(i, p) = side;
        }
    }
}

int ::EnsembleSpaceBuilder::set_shared_faces(imat_wrapper &h_shared_faces) const
{
    const int n = shared_faces.size();
    cuddh_verify(h_shared_faces.shape(0) == 4 && h_shared_faces.shape(1) == n,
                 printf("EnsembleSpace error: invalid shared_faces output shape."));

    for (int i = 0; i < n; ++i)
    {
        h_shared_faces(0, i) = shared_faces.at(i)[0];
        h_shared_faces(1, i) = shared_faces.at(i)[1];
        h_shared_faces(2, i) = shared_faces.at(i)[2];
        h_shared_faces(3, i) = shared_faces.at(i)[3];
    }

    return n;
}

void ::EnsembleSpaceBuilder::compute_fdof_indices(TensorWrapper<3, int> &h_fI, const TensorWrapper<4, int> &h_sI) const
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

            const EdgeConnectivity edge = mesh.edge_connectivity(g_f);
            const int el = el2s(edge.elements[side]);

            for (int i = 0; i < n_basis; ++i)
            {
                const int j = permute_edge_index(n_basis, i, edge.permutation);
                const auto [m, n] = edge2vol(n_basis, j, edge.labels[side]);

                const int idx = h_sI(m, n, el, p);

                h_fI(i, f, p) = idx;
            }
        }
    }
}

int ::EnsembleSpaceBuilder::set_subdomain_num_dofs(ivec_wrapper &h_s_dof) const
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

int ::EnsembleSpaceBuilder::set_subdomain_num_fdofs(ivec_wrapper &h_s_fdof) const
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

void ::EnsembleSpaceBuilder::compute_dof_indices(TensorWrapper<4, int> &h_sI) const
{
    const int n_spaces = E.size();
    const int n_basis = fem.basis().size();

    auto g_inds = fem.global_indices(MemorySpace::HOST); // global element indices

    for (int p = 0; p < n_spaces; ++p)
    {
        std::unordered_map<int, int> inv_indices; // global index to subspace index

        auto &dof_indices = s2g.at(p);
        auto &subdomain_elements = E.at(p);

        const int n_elem = subdomain_elements.size();
        const int ndof = dof_indices.size();

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

void ::EnsembleSpaceBuilder::compute_global_indices(imat_wrapper &h_gI) const
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

template <typename scalar_t, bool Complex>
void EnsembleSpace::set_pattern(BlockSparseMatrix<scalar_t, Complex> &B) const
{
    const auto sI = subspace_indices(MemorySpace::HOST);
    const auto nel = n_elems(MemorySpace::HOST);

    for (int p = 0; p < n_spaces; ++p)
        for (int el = 0; el < nel(p); ++el)
            for (int ia = 0; ia < n_basis; ++ia)
                for (int ja = 0; ja < n_basis; ++ja)
                {
                    const int row = sI(ia, ja, el, p);
                    for (int ib = 0; ib < n_basis; ++ib)
                        for (int jb = 0; jb < n_basis; ++jb)
                            B.add_entry(p, row, sI(ib, jb, el, p));
                }
}

template void EnsembleSpace::set_pattern(BlockSparseMatrix<float, false> &) const;
template void EnsembleSpace::set_pattern(BlockSparseMatrix<float, true> &) const;
template void EnsembleSpace::set_pattern(BlockSparseMatrix<double, false> &) const;
template void EnsembleSpace::set_pattern(BlockSparseMatrix<double, true> &) const;
