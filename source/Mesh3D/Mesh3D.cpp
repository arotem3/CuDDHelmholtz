#include "Mesh3D/Mesh3D.hpp"

#include <fstream>

using namespace cuddh;

using face_nodes = std::array<int, 4>;

static FaceConnectivity::Permutation compute_face_permutation(const face_nodes &owner, const face_nodes &neighbor)
{
    // Check permutations for the 8 possible rotations and reflections
    if (owner == neighbor)
    {
        return FaceConnectivity::Permutation::Identity; // Identity (no rotation, no mirroring)
    }

    // 90-degree rotation (clockwise)
    if ((owner[0] == neighbor[3] &&
         owner[1] == neighbor[0] &&
         owner[2] == neighbor[1] &&
         owner[3] == neighbor[2]))
    {
        return FaceConnectivity::Permutation::Rotate90;
    }

    // 180-degree rotation
    if ((owner[0] == neighbor[2] &&
         owner[1] == neighbor[3] &&
         owner[2] == neighbor[0] &&
         owner[3] == neighbor[1]))
    {
        return FaceConnectivity::Permutation::Rotate180;
    }

    // 270-degree rotation (clockwise)
    if ((owner[0] == neighbor[1] &&
         owner[1] == neighbor[2] &&
         owner[2] == neighbor[3] &&
         owner[3] == neighbor[0]))
    {
        return FaceConnectivity::Permutation::Rotate270;
    }

    // Horizontal flip
    if ((owner[0] == neighbor[1] &&
         owner[1] == neighbor[0] &&
         owner[2] == neighbor[3] &&
         owner[3] == neighbor[2]))
    {
        return FaceConnectivity::Permutation::HorizontalFlip;
    }

    // Vertical flip
    if ((owner[0] == neighbor[2] &&
         owner[1] == neighbor[3] &&
         owner[2] == neighbor[0] &&
         owner[3] == neighbor[1]))
    {
        return FaceConnectivity::Permutation::VerticalFlip;
    }

    // Main diagonal flip
    if ((owner[0] == neighbor[3] &&
         owner[1] == neighbor[0] &&
         owner[2] == neighbor[1] &&
         owner[3] == neighbor[2]))
    {
        return FaceConnectivity::Permutation::DiagonalFlipMain;
    }

    // Anti-diagonal flip
    if ((owner[0] == neighbor[2] &&
         owner[1] == neighbor[3] &&
         owner[2] == neighbor[0] &&
         owner[3] == neighbor[1]))
    {
        return FaceConnectivity::Permutation::DiagonalFlipAnti;
    }

    // If no match is found, throw an error
    throw std::runtime_error("compute_face_permutation: Face node permutation not recognized.");
}

class face_nodes_hash // hash function for face_nodes independent of permutation
{
public:
    size_t operator()(face_nodes a) const
    {
        std::sort(a.begin(), a.end());

        size_t h = 0;
        for (int x : a)
        {
            h ^= std::hash<int>{}(x) + 0x9e3779b9 + (h << 6) + (h >> 2);
        }
        return h;
    }
};

class face_nodes_equal // equality function for std::array<int, 4> independent of permutation
{
public:
    bool operator()(face_nodes a, face_nodes b) const
    {
        std::sort(a.begin(), a.end());
        std::sort(b.begin(), b.end());
        return a == b;
    }
};

static double dist3(double3 a, double3 b)
{
    return std::sqrt((a.x - b.x) * (a.x - b.x) +
                     (a.y - b.y) * (a.y - b.y) +
                     (a.z - b.z) * (a.z - b.z));
}

Mesh3D Mesh3D::uniform_cube(int nx, double ax, double bx, int ny, double ay, double by, int nz, double az, double bz)
{
    int np = (nx + 1) * (ny + 1) * (nz + 1);
    int nel = nx * ny * nz;

    Cube<double3> coo(nx + 1, ny + 1, nz + 1);
    Tensor<4, int> elems(8, nx, ny, nz);

    auto index = [=](int i, int j, int k) -> int
    {
        return i + (nx + 1) * (j + (ny + 1) * k);
    };

    double dx = (bx - ax) / nx;
    double dy = (by - ay) / ny;
    double dz = (bz - az) / nz;

    for (int k = 0; k <= nz; ++k)
    {
        const double z = az + dz * k;
        for (int j = 0; j <= ny; ++j)
        {
            const double y = ay + dy * j;
            for (int i = 0; i <= nx; ++i)
            {
                const double x = ax + dx * i;
                coo(i, j, k) = double3{x, y, z};
            }
        }
    }

    for (int k = 0; k < nz; ++k)
    {
        for (int j = 0; j < ny; ++j)
        {
            for (int i = 0; i < nx; ++i)
            {
                elems(0, i, j, k) = index(i, j, k);
                elems(1, i, j, k) = index(i + 1, j, k);
                elems(2, i, j, k) = index(i + 1, j + 1, k);
                elems(3, i, j, k) = index(i, j + 1, k);
                elems(4, i, j, k) = index(i, j, k + 1);
                elems(5, i, j, k) = index(i + 1, j, k + 1);
                elems(6, i, j, k) = index(i + 1, j + 1, k + 1);
                elems(7, i, j, k) = index(i, j + 1, k + 1);
            }
        }
    }

    return from_vertices(np, coo, nel, elems);
}

Mesh3D Mesh3D::from_vertices(int nx, const double3 *nodes, int nel, const int *elems)
{
    auto elem = reshape(elems, 8, nel);

    Mesh3D mesh;
    mesh.nel = nel;

    // initialize nodes and elements-to-nodes connectivity
    mesh.nodes.resize(nx);
    std::copy_n(nodes, nx, mesh.nodes.host_write());

    mesh.elems.resize(8 * nel);
    std::copy_n(elems, 8 * nel, mesh.elems.host_write());

    // compute mesh size
    double h = INFINITY;
    for (int el = 0; el < nel; ++el)
    {
        for (int i = 0; i < 8; ++i)
        {
            for (int j = i + 1; j < 8; ++j)
            {
                double3 x0 = nodes[elem(i, el)];
                double3 x1 = nodes[elem(j, el)];
                double d = dist3(x0, x1);
                h = std::min(h, d);
            }
        }
    }
    mesh._h = h;

    std::unordered_map<face_nodes, FaceConnectivity, face_nodes_hash, face_nodes_equal> face_map;
    face_map.reserve(6 * nel);

    int n_interior_faces = 0;

    constexpr int hex_faces[6][4] = {
        {0, 1, 2, 3}, // ZMin
        {4, 5, 6, 7}, // ZMax
        {0, 3, 7, 4}, // XMin
        {1, 2, 6, 5}, // XMax
        {0, 1, 5, 4}, // YMin
        {3, 2, 6, 7}  // YMax
    };

    // identify unique faces
    for (int el = 0; el < nel; ++el)
    {
        for (int f = 0; f < 6; ++f)
        {
            face_nodes face = {
                elem(hex_faces[f][0], el),
                elem(hex_faces[f][1], el),
                elem(hex_faces[f][2], el),
                elem(hex_faces[f][3], el)};

            auto it = face_map.find(face);
            if (it == face_map.end())
            {
                FaceConnectivity connectivity;

                connectivity.elements[0] = el;
                connectivity.elements[1] = -1;
                connectivity.label[0] = static_cast<FaceConnectivity::Label>(f);
                connectivity.label[1] = FaceConnectivity::Label::None;
                connectivity.permutation = FaceConnectivity::Permutation::Identity;

                face_map[face] = connectivity;
            }
            else
            {
                const auto &owner = it->first;
                auto &connectivity = it->second;

                if (connectivity.elements[1] >= 0)
                {
                    throw std::runtime_error("Mesh3D::from_vertices: Invalid mesh detected. More than two elements share the same face.");
                }

                connectivity.elements[1] = el;
                connectivity.label[1] = static_cast<FaceConnectivity::Label>(f);
                connectivity.permutation = compute_face_permutation(owner, face);

                n_interior_faces++;
            }
        }
    }

    const int n_faces = face_map.size();
    const int n_boundary_faces = n_faces - n_interior_faces;
    mesh.nf = n_faces;
    mesh.nbf = n_boundary_faces;
    mesh.nif = n_interior_faces;

    // classify faces
    mesh.interior_faces.resize(n_interior_faces);
    auto interior_faces = reshape(mesh.interior_faces.host_write(), n_interior_faces);

    mesh.boundary_faces.resize(n_boundary_faces);
    auto boundary_faces = reshape(mesh.boundary_faces.host_write(), n_boundary_faces);

    mesh.faces.resize(4 * n_faces);
    auto faces = reshape(mesh.faces.host_write(), 4, n_faces);

    mesh.connectivity.resize(n_faces);

    int l = 0, I = 0, B = 0;
    for (auto &[face, connectivity] : face_map)
    {
        for (int i = 0; i < 4; ++i)
            faces(i, l) = face[i];

        const auto [e0, e1] = connectivity.elements;

        if (e1 < 0) // boundary
        {
            boundary_faces[B] = l;
            B++;
        }
        else
        {
            interior_faces[I] = l;
            I++;
        }

        mesh.connectivity[l] = connectivity;
        l++;
    }

    assert(I == n_interior_faces);
    assert(B == n_boundary_faces);

    return mesh;
}

DeviceMesh3D Mesh3D::to_device() const
{
    DeviceMesh3D d_mesh;

    d_mesh.nodes = reshape(nodes.device_read(), nodes.size());
    d_mesh.elems = reshape(elems.device_read(), 8, nel);
    d_mesh.faces = reshape(faces.device_read(), 4, nf);
    d_mesh.interior_faces = reshape(interior_faces.device_read(), nif);
    d_mesh.boundary_faces = reshape(boundary_faces.device_read(), nbf);

    return d_mesh;
}
