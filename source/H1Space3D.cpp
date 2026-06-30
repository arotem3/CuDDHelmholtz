#include "FEM3D/H1Space3D.hpp"

using namespace cuddh;

H1Space3D::H1Space3D(const Mesh3D &mesh, const Basis &basis)
    : n_elem(mesh.n_elem()), n_basis(basis.size()), _mesh(mesh), _basis(basis), _I(n_basis * n_basis * n_basis * n_elem)
{
    auto I = reshape(_I.host_write(), n_basis, n_basis, n_basis, n_elem);

    std::unordered_map<int, int> duplicate_dof_map; // maps global DOF index to unique representative DOF index

    auto canonical = [&duplicate_dof_map](int x) -> int {
        while (duplicate_dof_map.contains(x))
            x = duplicate_dof_map.at(x);
        return x;
    };

    // iterate over faces
    const int n_faces = mesh.n_interior_faces();
    for (int f = 0; f < n_faces; ++f)
    {
        const FaceConnectivity fc = mesh.interior_face_connectivity(f);

        const auto [el0, el1] = fc.elements;
        const auto [f0, f1] = fc.label;

        for (int i = 0; i < n_basis; ++i)
        {
            for (int j = 0; j < n_basis; ++j)
            {
                auto vol_idx = face2vol(n_basis, i, j, f0);
                const int idx0 = vol_idx[0] + n_basis * (vol_idx[1] + n_basis * (vol_idx[2] + n_basis * el0));

                const auto [i1, j1] = permute_face_index(n_basis, i, j, fc.permutation);
                vol_idx = face2vol(n_basis, i1, j1, f1);
                const int idx1 = vol_idx[0] + n_basis * (vol_idx[1] + n_basis * (vol_idx[2] + n_basis * el1));

                const int c0 = canonical(idx0);
                const int c1 = canonical(idx1);
                if (c0 != c1)
                    duplicate_dof_map.insert({c1, c0});
            }
        }
    }

    // fill the global indices
    const int N = n_basis * n_basis * n_basis * n_elem;
    ndof = N - duplicate_dof_map.size();
    int l = 0;

    for (int i = 0; i < N; ++i)
    {
        if (not duplicate_dof_map.contains(i))
        {
            I[i] = l;
            ++l;
        }
        else
        {
            I[i] = -1;
        }
    }

    for (const auto [idx1, idx0] : duplicate_dof_map)
        I[idx1] = I[canonical(idx0)];

    // fill the physical coordinates
    _xyz.resize(ndof);
    auto xyz = reshape(_xyz.host_write(), ndof);

    auto q = _basis.quadrature().x(MemorySpace::HOST);

    for (int el = 0; el < n_elem; ++el)
    {
        const HexElement elem = mesh.element(el);

        for (int i = 0; i < n_basis; ++i)
        {
            for (int j = 0; j < n_basis; ++j)
            {
                for (int k = 0; k < n_basis; ++k)
                {
                    const double3 r{q(i), q(j), q(k)};
                    const int idx = I(i, j, k, el);
                    xyz(idx) = elem.physical_coordinates(r);
                }
            }
        }
    }
}

TraceSpace3D::TraceSpace3D(const H1Space3D &fem, int n_faces, const int *faces)
    : fem{fem}, nf{n_faces}, n_basis{fem.basis().size()}, _I(n_basis * n_basis * n_faces), _faces(n_faces)
{
    auto F = reshape(_faces.host_write(), n_faces);
    auto I = reshape(_I.host_write(), n_basis, n_basis, n_faces);

    for (int i = 0; i < n_faces; ++i)
    {
        F[i] = faces[i];
    }

    const Mesh3D &mesh = fem.mesh();
    const int n_elem = mesh.n_elem();
    auto K = reshape(fem.global_indices(MemorySpace::HOST), n_basis, n_basis, n_basis, n_elem);

    std::unordered_map<int, int> global_to_trace;
    std::vector<int> P; // global DOFs corresponding to trace DOFs

    global_to_trace.reserve(n_basis * n_basis * n_faces);
    P.reserve(n_basis * n_basis * n_faces);

    int l = 0;
    for (int f = 0; f < n_faces; ++f)
    {
        const FaceConnectivity connectivity = fem.mesh().face_connectivity(F[f]);

        for (int i = 0; i < n_basis; ++i)
        {
            for (int j = 0; j < n_basis; ++j)
            {
                const auto vol_idx = face2vol(n_basis, i, j, connectivity.label[0]);
                const int idx = K(vol_idx[0], vol_idx[1], vol_idx[2], connectivity.elements[0]);

                if (not global_to_trace.contains(idx))
                {
                    global_to_trace[idx] = l;
                    P.push_back(idx);
                    ++l;
                }

                I(i, j, f) = global_to_trace[idx];
            }
        }
    }

    ndof = global_to_trace.size();

    _proj.resize(ndof);
    auto proj = _proj.host_write();
    for (int i = 0; i < ndof; ++i)
    {
        proj[i] = P[i];
    }
}

void TraceSpace3D::restrict(const double *x, double *y) const
{
    auto proj = global_indices(MemorySpace::DEVICE);

    forall(ndof, [=] __device__(int i) -> void { y[i] = x[proj[i]]; });
}

void TraceSpace3D::prolong(const double *x, double *y) const
{
    auto proj = global_indices(MemorySpace::DEVICE);

    forall(ndof, [=] __device__(int i) -> void { y[proj[i]] += x[i]; });
}

void TraceSpace3D::orth(double *x) const
{
    auto proj = global_indices(MemorySpace::DEVICE);

    forall(ndof, [=] __device__(int i) -> void { x[proj[i]] = 0.0; });
}

void H1Space3D::set_pattern(SparseMatrix<double> &S) const
{
    auto I = global_indices(MemorySpace::HOST);
    for (int el = 0; el < n_elem; ++el)
        for (int a = 0; a < n_basis; ++a)
            for (int b = 0; b < n_basis; ++b)
                for (int cv = 0; cv < n_basis; ++cv)
                {
                    int row = I(a, b, cv, el);
                    for (int d = 0; d < n_basis; ++d)
                        for (int e = 0; e < n_basis; ++e)
                            for (int f = 0; f < n_basis; ++f)
                                S.add_entry(row, I(d, e, f, el));
                }
}

void H1Space3D::set_pattern(SparseMatrix<double, true> &S) const
{
    auto I = global_indices(MemorySpace::HOST);
    for (int el = 0; el < n_elem; ++el)
        for (int a = 0; a < n_basis; ++a)
            for (int b = 0; b < n_basis; ++b)
                for (int cv = 0; cv < n_basis; ++cv)
                {
                    int row = I(a, b, cv, el);
                    for (int d = 0; d < n_basis; ++d)
                        for (int e = 0; e < n_basis; ++e)
                            for (int f = 0; f < n_basis; ++f)
                                S.add_entry(row, I(d, e, f, el));
                }
}

