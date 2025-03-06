#include "H1Space3D.hpp"

using namespace cuddh;

template <typename Map, typename Key>
static bool contains(const Map &map, Key key)
{
    return map.find(key) != map.end();
}

static constexpr auto permute(int N, int i, int j, FaceConnectivity::Permutation p)
{
    switch (p)
    {
    case FaceConnectivity::Permutation::Rotate90:
        return std::make_pair(j, N - 1 - i);
    case FaceConnectivity::Permutation::Rotate180:
        return std::make_pair(N - 1 - i, N - 1 - j);
    case FaceConnectivity::Permutation::Rotate270:
        return std::make_pair(N - 1 - j, i);
    case FaceConnectivity::Permutation::HorizontalFlip:
        return std::make_pair(N - 1 - i, j);
    case FaceConnectivity::Permutation::VerticalFlip:
        return std::make_pair(i, N - 1 - j);
    case FaceConnectivity::Permutation::DiagonalFlipMain:
        return std::make_pair(j, i);
    case FaceConnectivity::Permutation::DiagonalFlipAnti:
        return std::make_pair(N - 1 - j, N - 1 - i);
    default: // Identity
        return std::make_pair(i, j);
    }
}

static constexpr int face2vol(int N, int i, int j, FaceConnectivity::Label f, int el)
{
    int m = 0, n = 0, l = 0;

    if (f == FaceConnectivity::Label::ZMin || f == FaceConnectivity::Label::ZMax)
    {
        m = i;
        n = j;
        l = (f == FaceConnectivity::Label::ZMin) ? 0 : (N - 1);
    }
    else if (f == FaceConnectivity::Label::XMin || f == FaceConnectivity::Label::XMax)
    {
        m = (f == FaceConnectivity::Label::XMin) ? 0 : (N - 1);
        n = i;
        l = j;
    }
    else // YMin or YMax
    {
        m = i;
        n = (f == FaceConnectivity::Label::YMin) ? 0 : (N - 1);
        l = j;
    }

    return m + N * (n + N * (l + N * el));
}

H1Space3D::H1Space3D(const Mesh3D &mesh, const Basis &basis)
    : n_elem(mesh.n_elem()),
      n_basis(basis.size()),
      _mesh(mesh),
      _basis(basis),
      _I(n_basis * n_basis * n_basis * n_elem)
{
    auto I = reshape(_I.host_write(), n_basis, n_basis, n_basis, n_elem);

    std::unordered_map<int, int> mask;

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
                const int idx0 = face2vol(n_basis, i, j, f0, el0);

                const auto [i1, j1] = permute(n_basis, i, j, fc.permutation);
                const int idx1 = face2vol(n_basis, i1, j1, f1, el1);

                int i = contains(mask, idx0) ? mask[idx0] : idx0;
                mask.insert({idx1, i});
            }
        }
    }

    // fill the global indices
    const int N = n_basis * n_basis * n_basis * n_elem;
    ndof = N - mask.size();
    int l = 0;

    for (int i = 0; i < N; ++i)
    {
        if (not contains(mask, i))
        {
            I[i] = l;
            ++l;
        }
        else
        {
            I[i] = -1;
        }
    }

    for (const auto [idx1, idx0] : mask)
    {
        int i = idx0;

        const int safety_factor = 10; // Arbitrary safety factor to prevent infinite loop
        int count = 0;
        while (contains(mask, i))
        {
            if (++count > safety_factor)
                throw std::runtime_error("H1Space3D: Infinite loop detected in connectivity graph. Possible degenerate mesh.");
            
            i = mask[i];
        }

        I[idx1] = I[i];
    }

    // fill the physical coordinates
    _xyz.resize(ndof);
    auto xyz = reshape(_xyz.host_write(), ndof);

    for (int el = 0; el < n_elem; ++el)
    {
        const HexElement elem = mesh.element(el);

        for (int i = 0; i < n_basis; ++i)
        {
            for (int j = 0; j < n_basis; ++j)
            {
                for (int k = 0; k < n_basis; ++k)
                {
                    const double3 r = double3{_basis.quadrature().x(i), _basis.quadrature().x(j), _basis.quadrature().x(k)};
                    const int idx = I(i, j, k, el);

                    xyz(idx) = elem.physical_coordinates(r);
                }
            }
        }
    }
}

TraceSpace3D::TraceSpace3D(const H1Space3D &fem, int n_faces, const int *faces)
    : fem{fem},
     nf{n_faces},
     n_basis{fem.basis().size()},
     _I(n_basis * n_basis * n_faces),
     _faces(n_faces)
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

    std::unordered_map<int, int> mask; // unique mapping from global DOFs to trace DOFs
    std::vector<int> P;               // global DOFs corresponding to trace DOFs

    mask.reserve(n_basis * n_basis * n_faces);
    P.reserve(n_basis * n_basis * n_faces);

    int l = 0;
    for (int f = 0; f < n_faces; ++f)
    {
        const FaceConnectivity connectivity = fem.mesh().interior_face_connectivity(F[f]);

        const int el = connectivity.elements[0];
        const FaceConnectivity::Label s = connectivity.label[0];

        for (int i = 0; i < n_basis; ++i)
        {
            for (int j = 0; j < n_basis; ++j)
            {
                const int idx = face2vol(n_basis, i, j, s, el);

                if (not contains(mask, idx))
                {
                    mask[idx] = l;
                    P.push_back(idx);
                    ++l;
                }

                I(i, j, f) = mask[idx];
            }
        }
    }

    ndof = mask.size();

    _proj.resize(ndof);
    auto proj = _proj.host_write();
    for (int i = 0; i < ndof; ++i)
    {
        proj[i] = P[i];
    }
}

void TraceSpace3D::restrict(const double * x, double * y) const
{
    auto proj = global_indices(MemorySpace::DEVICE);

    forall(ndof, [=] __device__ (int i) -> void
    {
        y[i] = x[proj[i]];
    });
}

void TraceSpace3D::prolong(const double * x, double * y) const
{
    auto proj = global_indices(MemorySpace::DEVICE);

    forall(ndof, [=] __device__ (int i) -> void
    {
        y[proj[i]] += x[i];
    });
}

void TraceSpace3D::orth(double * x) const
{
    auto proj = global_indices(MemorySpace::DEVICE);

    forall(ndof, [=] __device__ (int i) -> void
    {
        x[proj[i]] = 0.0;
    });
}
