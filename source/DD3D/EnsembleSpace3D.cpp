#include "DD3D/EnsembleSpace3D.hpp"

#include <algorithm>
#include <array>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

using namespace cuddh;

namespace
{
    class EnsembleSpaceBuilder
    {
    public:
        EnsembleSpaceBuilder(const H1Space3D &fem, int n_spaces, const int *element_labels);

        int set_subdomain_num_elements(ivec_wrapper &h_s_elems) const;
        void set_subdomain_elements(imat_wrapper &h_elems) const;
        int set_subdomain_num_boundary_faces(ivec_wrapper &h_s_faces) const;
        void set_subdomain_face_indices(imat_wrapper &h_faces) const;
        void set_subdomain_face_sides(imat_wrapper &h_face_sides) const;
        int n_shared_faces() const { return shared_faces.size(); }
        int set_shared_faces(imat_wrapper &h_shared_faces) const;
        void compute_fdof_indices(TensorWrapper<4, int> &h_fI, const TensorWrapper<5, int> &h_sI) const;
        int set_subdomain_num_dofs(ivec_wrapper &h_s_dof) const;
        int set_subdomain_num_fdofs(ivec_wrapper &h_s_fdof) const;
        void compute_dof_indices(TensorWrapper<5, int> &h_sI) const;
        void compute_global_indices(imat_wrapper &h_gI) const;

    private:
        const H1Space3D &fem;
        std::vector<std::vector<int>> E;                 // elements
        std::vector<std::vector<std::pair<int, int>>> F; // faces
        std::vector<std::array<int, 4>> shared_faces;    // {subdomain0, subdomain1, subdomain face index0, ..face..1}
        std::vector<std::vector<int>> s2g;               // subspace index to global index
        std::vector<std::vector<int>> f2s;               // face index to subspace index
    };
} // namespace

EnsembleSpace3D::EnsembleSpace3D(const H1Space3D &fem, int n_spaces, const int *element_labels)
    : fem(fem),
      n_spaces{n_spaces},
      n_basis{fem.basis().size()},
      sp_dof(n_spaces),
      sp_n_elems(n_spaces),
      sp_n_faces(n_spaces),
      fp_dof(n_spaces)
{
    auto &mesh = fem.mesh();
    const int nel = mesh.n_elem();

    auto h_s_elems = reshape(sp_n_elems.host_write(), n_spaces);
    auto h_s_faces = reshape(sp_n_faces.host_write(), n_spaces);
    auto h_s_dof = reshape(sp_dof.host_write(), n_spaces);
    auto h_s_fdof = reshape(fp_dof.host_write(), n_spaces);

    // determine elements in each subspace
    EnsembleSpaceBuilder ESbuilder(fem, n_spaces, element_labels);

    mx_elems = ESbuilder.set_subdomain_num_elements(h_s_elems);

    // map subspace element index to global element index
    sp_elems.resize(mx_elems * n_spaces);
    auto h_elems = reshape(sp_elems.host_write(), mx_elems, n_spaces);
    ESbuilder.set_subdomain_elements(h_elems);

    sp_element_labels.resize(nel);
    auto h_labels = reshape(sp_element_labels.host_write(), nel);
    for (int el = 0; el < nel; ++el)
        h_labels(el) = element_labels[el];

    // determine faces in each subspace
    mx_faces = ESbuilder.set_subdomain_num_boundary_faces(h_s_faces);

    // populate h_faces
    sp_faces.resize(mx_faces * n_spaces);
    auto h_faces = reshape(sp_faces.host_write(), mx_faces, n_spaces);
    ESbuilder.set_subdomain_face_indices(h_faces);

    sp_face_sides.resize(mx_faces * n_spaces);
    auto h_face_sides = reshape(sp_face_sides.host_write(), mx_faces, n_spaces);
    ESbuilder.set_subdomain_face_sides(h_face_sides);

    n_shared_faces = ESbuilder.n_shared_faces();
    sp_shared_faces.resize(4 * n_shared_faces);
    auto h_shared_faces = reshape(sp_shared_faces.host_write(), 4, n_shared_faces);
    ESbuilder.set_shared_faces(h_shared_faces);

    // determine mapping between global and subspace indices
    sp_indices.resize(n_basis * n_basis * n_basis * mx_elems * n_spaces);
    auto h_sI = reshape(sp_indices.host_write(), n_basis, n_basis, n_basis, mx_elems, n_spaces);
    ESbuilder.compute_dof_indices(h_sI);

    mx_ndof = ESbuilder.set_subdomain_num_dofs(h_s_dof);
    mx_fdof = ESbuilder.set_subdomain_num_fdofs(h_s_fdof);

    sp_global.resize(mx_ndof * n_spaces);
    auto h_gI = reshape(sp_global.host_write(), mx_ndof, n_spaces);
    ESbuilder.compute_global_indices(h_gI);

    fp_indices.resize(n_basis * n_basis * mx_faces * n_spaces);
    auto h_fI = reshape(fp_indices.host_write(), n_basis, n_basis, mx_faces, n_spaces);
    ESbuilder.compute_fdof_indices(h_fI, h_sI);
}

EnsembleSpace3D cuddh::partition_uniform_cube(const H1Space3D &fem, dim3 mesh_dims, dim3 block_dims)
{
    const int n_basis = fem.basis().size();
    const auto [nx, ny, nz] = mesh_dims;
    const auto [bx, by, bz] = block_dims;

    const int dx = (nx + bx - 1) / bx;
    const int dy = (ny + by - 1) / by;
    const int dz = (nz + bz - 1) / bz;

    int nd = dx * dy * dz;

    icube element_labels(nx, ny, nz);
    std::fill(element_labels.begin(), element_labels.end(), -1);

    for (int k = 0; k < nz; ++k)
    {
        for (int j = 0; j < ny; ++j)
        {
            for (int i = 0; i < nx; ++i)
            {
                int label_x = i / bx;
                int label_y = j / by;
                int label_z = k / bz;
                element_labels(i, j, k) = label_x + dx * (label_y + dy * label_z);
            }
        }
    }

    return EnsembleSpace3D(fem, nd, element_labels);
}

static auto compute_subspace_elements(int nel, int n_spaces, const int *element_labels)
{
    std::vector<std::vector<int>> E(n_spaces);

    for (int el = 0; el < nel; ++el)
    {
        const int p = element_labels[el];
        E.at(p).push_back(el);
    }

    return E;
}

static auto compute_subdomain_boundary_faces(const Mesh3D &mesh, int n_spaces, const int *element_labels)
{
    std::vector<std::vector<std::pair<int, int>>> F(n_spaces); // faces in each subspace
    std::vector<std::array<int, 4>> shared_faces; // {subdomain0, subdomain1, subdomain face index0, ..face..1}
    const int n_global_faces = mesh.n_faces();

    for (int face_index = 0; face_index < n_global_faces; ++face_index)
    {
        // loop over faces and check if a face is on the boundary of a
        // subdomain. Boundary faces are automatically on the boundary, and
        // interior faces are on the boundary only if the element[0] != element[1].

        const FaceConnectivity connectivity = mesh.face_connectivity(face_index);

        const auto [el0, el1] = connectivity.elements;
        const int domain0 = element_labels[el0];

        if (el1 < 0) // boundary
        {
            F.at(domain0).push_back({face_index, 0});
        }
        else
        {
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
    ivec el2s(nel);
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

::EnsembleSpaceBuilder::EnsembleSpaceBuilder(const H1Space3D &fem, int n_spaces, const int *element_labels) : fem(fem)
{
    const Mesh3D &mesh = fem.mesh();

    E = compute_subspace_elements(mesh.n_elem(), n_spaces, element_labels);
    std::tie(F, shared_faces) = compute_subdomain_boundary_faces(mesh, n_spaces, element_labels);

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
            const int g_el = subdomain_elements.at(el);
            for (int k = 0; k < n_basis; ++k)
            {
                for (int j = 0; j < n_basis; ++j)
                {
                    for (int i = 0; i < n_basis; ++i)
                    {
                        const int g_idx = g_inds(i, j, k, g_el); // global index
                        if (not unique.contains(g_idx))
                        {
                            unique[g_idx] = l;
                            dof_indices.push_back(g_idx);
                            ++l;
                        }
                    }
                }
            }
        }

        std::unordered_map<int, int> funique; // unique face indices
        auto &fdof_indices = f2s.at(p);
        auto &subdomain_faces = F.at(p);
        const int n_faces = subdomain_faces.size();

        l = 0;
        for (int f = 0; f < n_faces; ++f)
        {
            const auto [global_face, side] = subdomain_faces.at(f);
            const FaceConnectivity connectivity = mesh.face_connectivity(global_face);

            for (int j = 0; j < n_basis; ++j)
            {
                for (int i = 0; i < n_basis; ++i)
                {
                    int ip = i;
                    int jp = j;
                    if (side == 1)
                    {
                        const auto [ip1, jp1] = permute_face_index(n_basis, i, j, connectivity.permutation);
                        ip = ip1;
                        jp = jp1;
                    }

                    const auto vol_idx = face2vol(n_basis, ip, jp, connectivity.label[side]);
                    const int idx = unique.at(g_inds(vol_idx[0], vol_idx[1], vol_idx[2], connectivity.elements[side]));

                    if (not funique.contains(idx))
                    {
                        funique[idx] = l;
                        fdof_indices.push_back(idx);
                        ++l;
                    }
                }
            }
        }

        natural_ordering(dof_indices, fdof_indices);
    }
}

int ::EnsembleSpaceBuilder::set_subdomain_num_elements(ivec_wrapper &h_s_elems) const
{
    const int n_spaces = E.size();

    int mx = 0;

    for (int p = 0; p < n_spaces; ++p)
    {
        const int n = E.at(p).size();
        h_s_elems(p) = n;
        mx = std::max(mx, n);

        cuddh_verify(n >= 1, printf("EnsembleSpace3D error: Subspace %d is empty.\n", p));
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
        auto &faces = F.at(p);
        const int n = faces.size();
        for (int i = 0; i < n; ++i)
        {
            const int f = faces.at(i).first;
            h_faces(i, p) = f;
        }
    }
}

void ::EnsembleSpaceBuilder::set_subdomain_face_sides(imat_wrapper &h_face_sides) const
{
    const int n_spaces = E.size();

    std::fill(h_face_sides.begin(), h_face_sides.end(), -1);

    for (int p = 0; p < n_spaces; ++p)
    {
        auto &faces = F.at(p);
        const int n = faces.size();
        for (int i = 0; i < n; ++i)
        {
            const int side = faces.at(i).second;
            h_face_sides(i, p) = side;
        }
    }
}

int ::EnsembleSpaceBuilder::set_shared_faces(imat_wrapper &h_shared_faces) const
{
    const int n = shared_faces.size();
    if (h_shared_faces.shape(1) > 0)
    {
        cuddh_verify(h_shared_faces.shape(0) == 4 && h_shared_faces.shape(1) == n,
                     printf("EnsembleSpace3D error: invalid shared_faces output shape.\n"));

        for (int i = 0; i < n; ++i)
        {
            h_shared_faces(0, i) = shared_faces.at(i)[0];
            h_shared_faces(1, i) = shared_faces.at(i)[1];
            h_shared_faces(2, i) = shared_faces.at(i)[2];
            h_shared_faces(3, i) = shared_faces.at(i)[3];
        }
    }

    return n;
}

void ::EnsembleSpaceBuilder::compute_fdof_indices(TensorWrapper<4, int> &h_fI, const TensorWrapper<5, int> &h_sI) const
{
    const Mesh3D &mesh = fem.mesh();
    const Basis &basis = fem.basis();

    const int n_spaces = E.size();
    const int n_basis = basis.size();

    auto el2s = global_element_to_subspace_element(mesh.n_elem(), E);

    for (int p = 0; p < n_spaces; ++p)
    {
        auto &sf = F.at(p);
        const int nf = sf.size();

        for (int f = 0; f < nf; ++f)
        {
            const auto [global_face, side] = sf.at(f);
            const FaceConnectivity connectivity = mesh.face_connectivity(global_face);

            const int subsp_element = el2s[connectivity.elements[side]];

            for (int j = 0; j < n_basis; ++j)
            {
                for (int i = 0; i < n_basis; ++i)
                {
                    int ip = i;
                    int jp = j;
                    if (side == 1)
                    {
                        const auto [ip1, jp1] = permute_face_index(n_basis, i, j, connectivity.permutation);
                        ip = ip1;
                        jp = jp1;
                    }

                    const auto [x, y, z] = face2vol(n_basis, ip, jp, connectivity.label[side]);
                    const int idx = h_sI(x, y, z, subsp_element, p);

                    h_fI(i, j, f, p) = idx;
                }
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

void ::EnsembleSpaceBuilder::compute_dof_indices(TensorWrapper<5, int> &h_sI) const
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
            for (int k = 0; k < n_basis; ++k)
            {
                for (int j = 0; j < n_basis; ++j)
                {
                    for (int i = 0; i < n_basis; ++i)
                    {
                        const int g_idx = g_inds(i, j, k, g_el);
                        h_sI(i, j, k, el, p) = inv_indices.at(g_idx);
                    }
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
        auto &dof_indices = s2g.at(p);
        const int ndof = dof_indices.size();
        for (int i = 0; i < ndof; ++i)
        {
            h_gI(i, p) = dof_indices.at(i);
        }
    }
}
