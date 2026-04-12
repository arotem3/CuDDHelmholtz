#include "Mesh2D/Mesh2D.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_map>

namespace cuddh
{
    Mesh2D Mesh2D::from_vertices(int nx, const double *x_, int nel, const int *elems_)
    {
        auto coo = reshape(x_, 2, nx);
        auto elem_conn = reshape(elems_, 4, nel);

        constexpr int emap1[] = {0, 1, 3, 0};
        constexpr int emap2[] = {1, 2, 2, 3};

        Mesh2D mesh;

        // populate node coordinate array
        mesh._node_coords.resize(nx);
        double2 *nc = mesh._node_coords.host_write();
        for (int k = 0; k < nx; ++k)
            nc[k] = {coo(0, k), coo(1, k)};

        // populate elem connectivity
        mesh._elem_nodes.resize(4 * nel);
        auto ec = reshape(mesh._elem_nodes.host_write(), 4, nel);
        for (int el = 0; el < nel; ++el)
            for (int i = 0; i < 4; ++i)
                ec(i, el) = elem_conn(i, el);

        // construct edges
        std::unordered_map<int, int> edge_map;
        auto key = [nx](int i, int j) -> int {
            return std::min(i, j) + nx * std::max(i, j);
        };

        for (int el = 0; el < nel; ++el)
        {
            for (int s = 0; s < 4; ++s)
            {
                const int l1 = emap1[s];
                const int l2 = emap2[s];

                const int C0 = elem_conn(l1, el);
                const int C1 = elem_conn(l2, el);

                const int k = key(C0, C1);
                if (not edge_map.contains(k))
                {
                    EdgeConnectivity conn;
                    conn.elements[0] = el;
                    conn.elements[1] = -1;
                    conn.labels[0] = s;
                    conn.labels[1] = -1;
                    conn.permutation = 1;
                    mesh._edge_connectivity.push_back(conn);

                    edge_map[k] = (int)mesh._edge_connectivity.size() - 1;
                }
                else
                {
                    const int e = edge_map.at(k);
                    EdgeConnectivity &conn = mesh._edge_connectivity[e];

                    const int e0 = conn.elements[0];
                    const int s0 = conn.labels[0];
                    const int n1 = elem_conn(emap1[s0], e0);

                    conn.elements[1] = el;
                    conn.labels[1] = s;
                    conn.permutation = (C0 == n1) ? 1 : -1;
                }
            }
        }

        // classify edges
        std::vector<int> boundary_edge_ids, interior_edge_ids;
        for (int i = 0; i < (int)mesh._edge_connectivity.size(); ++i)
        {
            if (mesh._edge_connectivity[i].elements[1] == -1)
                boundary_edge_ids.push_back(i);
            else
                interior_edge_ids.push_back(i);
        }

        // populate edge arrays with correct ordering:
        // edge nodes are ordered so that the right-hand perpendicular
        // (rotation of edge direction by 90° clockwise) points outward.
        // For labels 2,3 (top and left edges), swap node indices for correct direction.
        const int ne = (int)mesh._edge_connectivity.size();
        mesh._edge_nodes.resize(2 * ne);
        auto en = reshape(mesh._edge_nodes.host_write(), 2, ne);
        for (int i = 0; i < ne; ++i)
        {
            const int side = mesh._edge_connectivity[i].labels[0];
            const int el = mesh._edge_connectivity[i].elements[0];
            const int n0 = elem_conn(emap1[side], el);
            const int n1 = elem_conn(emap2[side], el);
            // For labels 2,3 (top/left edges), swap to ensure right-perp points outward
            if (side == 2 || side == 3)
            {
                en(0, i) = n1;
                en(1, i) = n0;
            }
            else
            {
                en(0, i) = n0;
                en(1, i) = n1;
            }
        }

        const int n_int = (int)interior_edge_ids.size();
        mesh._interior_edges_d.resize(n_int);
        int *ie = mesh._interior_edges_d.host_write();
        for (int i = 0; i < n_int; ++i)
            ie[i] = interior_edge_ids[i];

        const int n_bnd = (int)boundary_edge_ids.size();
        mesh._boundary_edges_d.resize(n_bnd);
        int *be = mesh._boundary_edges_d.host_write();
        for (int i = 0; i < n_bnd; ++i)
            be[i] = boundary_edge_ids[i];

        return mesh;
    }

    Mesh2D Mesh2D::uniform_rect(int nx, double ax, double bx, int ny, double ay, double by)
    {
        int np = (nx + 1) * (ny + 1);
        int nel = nx * ny;
        dcube coo(2, nx + 1, ny + 1);
        Cube<int> elems(4, nx, ny);

        auto l = [nx](int i, int j) -> int {
            return i + (nx + 1) * j;
        };

        double dx = (bx - ax) / nx;
        double dy = (by - ay) / ny;
        for (int j = 0; j <= ny; ++j)
        {
            const double y = ay + dy * j;
            for (int i = 0; i <= nx; ++i)
            {
                coo(0, i, j) = ax + dx * i;
                coo(1, i, j) = y;
            }
        }

        for (int j = 0; j < ny; ++j)
        {
            for (int i = 0; i < nx; ++i)
            {
                elems(0, i, j) = l(i, j);
                elems(1, i, j) = l(i + 1, j);
                elems(2, i, j) = l(i + 1, j + 1);
                elems(3, i, j) = l(i, j + 1);
            }
        }

        return from_vertices(np, coo, nel, elems);
    }

    QuadElement Mesh2D::element(int el) const
    {
        cuddh_assert(0 <= el && el < n_elem(),
                     printf("Mesh2D error: element el = %d out of range (#elements = %d)\n", el, n_elem()));
        auto ec = reshape(_elem_nodes.host_read(), 4, n_elem());
        const double2 *nc = _node_coords.host_read();
        double2 corners[4];
        for (int i = 0; i < 4; ++i)
            corners[i] = nc[ec(i, el)];
        return QuadElement(corners);
    }

    Edge Mesh2D::edge(int i) const
    {
        cuddh_assert(
            0 <= i && i < (int)_edge_connectivity.size(),
            printf("Mesh2D error: edge i = %d out of range (#edges = %d)\n", i, (int)_edge_connectivity.size()));
        const double2 *nc = _node_coords.host_read();
        auto en = reshape(_edge_nodes.host_read(), 2, n_edges());
        return Edge(nc[en(0, i)], nc[en(1, i)]);
    }

    Edge Mesh2D::edge(int i, FaceType type) const
    {
        if (type == FaceType::BOUNDARY)
        {
            cuddh_assert(0 <= i && i < _boundary_edges_d.size(),
                         printf("Mesh2D error: boundary edge i = %d out of range (#boundary edges = %d)\n", i,
                                _boundary_edges_d.size()));
            return edge(_boundary_edges_d.host_read()[i]);
        }
        else
        {
            cuddh_assert(0 <= i && i < _interior_edges_d.size(),
                         printf("Mesh2D error: interior edge i = %d out of range (#interior edges = %d)\n", i,
                                _interior_edges_d.size()));
            return edge(_interior_edges_d.host_read()[i]);
        }
    }

    EdgeConnectivity Mesh2D::edge_connectivity(int i) const
    {
        cuddh_assert(
            0 <= i && i < (int)_edge_connectivity.size(),
            printf("Mesh2D error: edge i = %d out of range (#edges = %d)\n", i, (int)_edge_connectivity.size()));
        return _edge_connectivity[i];
    }

    EdgeConnectivity Mesh2D::edge_connectivity(int i, FaceType type) const
    {
        if (type == FaceType::BOUNDARY)
        {
            cuddh_assert(0 <= i && i < _boundary_edges_d.size(),
                         printf("Mesh2D error: boundary edge i = %d out of range (#boundary edges = %d)\n", i,
                                _boundary_edges_d.size()));
            return _edge_connectivity[_boundary_edges_d.host_read()[i]];
        }
        else
        {
            cuddh_assert(0 <= i && i < _interior_edges_d.size(),
                         printf("Mesh2D error: interior edge i = %d out of range (#interior edges = %d)\n", i,
                                _interior_edges_d.size()));
            return _edge_connectivity[_interior_edges_d.host_read()[i]];
        }
    }

    ivec Mesh2D::boundary_edges() const
    {
        const int ne = _boundary_edges_d.size();
        ivec b(ne);
        const int *be = _boundary_edges_d.host_read();
        for (int i = 0; i < ne; ++i)
            b(i) = be[i];
        return b;
    }

    DeviceMesh2D Mesh2D::to_device() const
    {
        DeviceMesh2D d;
        d.nodes = reshape(_node_coords.device_read(), _node_coords.size());
        d.elems = reshape(_elem_nodes.device_read(), 4, n_elem());
        d.edge_nodes = reshape(_edge_nodes.device_read(), 2, n_edges());
        d.interior_edges = reshape(_interior_edges_d.device_read(), n_edges(FaceType::INTERIOR));
        d.boundary_edges = reshape(_boundary_edges_d.device_read(), n_edges(FaceType::BOUNDARY));
        return d;
    }

    double Mesh2D::h() const
    {
        double h = std::numeric_limits<double>::infinity();
        const double2 *nc = _node_coords.host_read();
        auto en = reshape(_edge_nodes.host_read(), 2, n_edges());
        for (int i = 0; i < n_edges(); ++i)
        {
            double2 a = nc[en(0, i)], b = nc[en(1, i)];
            double dx = b.x - a.x, dy = b.y - a.y;
            h = std::min(h, sqrt(dx * dx + dy * dy));
        }
        return h;
    }
} // namespace cuddh
