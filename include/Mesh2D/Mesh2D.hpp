#ifndef CUDDH_MESH_2D_HPP
#define CUDDH_MESH_2D_HPP

#include <vector>

#include "Edge.hpp"
#include "Element.hpp"
#include "HostDeviceArray.hpp"
#include "Tensor.hpp"

namespace cuddh
{
    class DeviceMesh2D;

    /// @brief The 2D mesh.
    class Mesh2D
    {
    public:
        /// @brief constructs empty mesh
        Mesh2D() = default;
        ~Mesh2D() = default;

        Mesh2D(const Mesh2D &) = delete;
        Mesh2D &operator=(const Mesh2D &) = delete;

        Mesh2D(Mesh2D &&) = default;
        Mesh2D &operator=(Mesh2D &&) = default;

        /// @brief number of elements in mesh.
        int n_elem() const { return _elem_nodes.size() / 4; }

        /// @brief number of mesh vertices.
        int n_nodes() const { return _node_coords.size(); }

        /// @brief total number of edges in mesh.
        int n_edges() const { return _edge_nodes.size() / 2; }

        /// @brief number of edges of the specified type.
        int n_edges(FaceType type) const
        {
            return (type == FaceType::BOUNDARY) ? _boundary_edges_d.size() : _interior_edges_d.size();
        }

        /// @brief returns the shortest edge length in the mesh.
        double h() const;

        /// @brief Returns the geometry of edge i (by global edge index).
        Edge edge(int i) const;

        /// @brief Returns the geometry of the i-th edge of the specified type.
        Edge edge(int i, FaceType type) const;

        /// @brief Returns the connectivity of edge i (by global edge index).
        EdgeConnectivity edge_connectivity(int i) const;

        /// @brief Returns the connectivity of the i-th edge of the specified type.
        EdgeConnectivity edge_connectivity(int i, FaceType type) const;

        /// @brief Returns the indices of the boundary edges (global edge indices).
        ivec boundary_edges() const;

        /// @brief Returns the geometry of element el.
        QuadElement element(int el) const;

        /// @brief Returns read-only host access to the element-node connectivity
        /// array, shaped (4, n_elem()): ec(i, el) is the global node index of
        /// corner i of element el.
        const_imat_wrapper elem_nodes() const { return reshape(_elem_nodes.host_read(), 4, n_elem()); }

        /// @brief Returns a device-accessible view of the mesh geometry.
        DeviceMesh2D to_device() const;

        /// @brief Constructs a mesh of QuadElements from a list of vertices and
        /// element corner indices.
        /// @param[in] nx number of vertices
        /// @param[in] x  shape (2, nx): vertex coordinates
        /// @param[in] nel number of elements
        /// @param[in] elems shape (4, nel): corner indices in CCW order
        static Mesh2D from_vertices(int nx, const double *x, int nel, const int *elems);

        /// @brief Constructs a uniform structured mesh for [ax,bx] x [ay,by]
        /// with nx * ny quadrilateral elements.
        static Mesh2D uniform_rect(int nx, double ax, double bx, int ny, double ay, double by);

    private:
        std::vector<EdgeConnectivity> _edge_connectivity; ///< edge topology, indexed by global edge id

        HostDeviceArray<double2> _node_coords;  ///< (n_nodes,)       device-accessible node positions
        HostDeviceArray<int> _elem_nodes;       ///< (4*n_elem,)      device-accessible element connectivity
        HostDeviceArray<int> _edge_nodes;       ///< (2*n_edges,)     device-accessible edge node indices (pre-ordered for outward normal)
        HostDeviceArray<int> _interior_edges_d; ///< (n_interior,)    device-accessible interior edge indices
        HostDeviceArray<int> _boundary_edges_d; ///< (n_boundary,)    device-accessible boundary edge indices
    };

    /// @brief A 2D mesh of quadrilateral elements on the device.
    ///
    /// Contains device pointers to node coordinates, element and edge node
    /// connectivity, allowing CUDA kernels to reconstruct element and edge
    /// metrics on the fly. Obtained via `Mesh2D::to_device()`.
    class DeviceMesh2D
    {
    public:
        DeviceMesh2D() = default;
        DeviceMesh2D(const DeviceMesh2D &) = default;
        DeviceMesh2D(DeviceMesh2D &&) = default;
        DeviceMesh2D &operator=(const DeviceMesh2D &) = default;
        DeviceMesh2D &operator=(DeviceMesh2D &&) = default;

        __host__ __device__ int n_elem() const { return elems.shape(1); }
        __host__ __device__ int n_edges() const { return edge_nodes.shape(1); }
        __host__ __device__ int n_interior_edges() const { return interior_edges.size(); }
        __host__ __device__ int n_boundary_edges() const { return boundary_edges.size(); }

        /// @brief Reconstructs element el from device node and connectivity data.
        __device__ QuadElement element(int el) const
        {
            cuddh_assert(0 <= el && el < n_elem(),
                         printf("DeviceMesh2D error: element index %d out of range [0, %d).\n", el, n_elem()););
            double2 corners[4];
            for (int i = 0; i < 4; ++i)
                corners[i] = nodes[elems(i, el)];
            return QuadElement(corners);
        }

        /// @brief Reconstructs the geometry of edge e (by global edge index) on the fly.
        __device__ Edge edge(int e) const
        {
            return Edge(nodes[edge_nodes(0, e)], nodes[edge_nodes(1, e)]);
        }

        /// @brief Returns the geometry of interior edge e.
        __device__ Edge interior_edge(int e) const { return edge(interior_edges[e]); }

        /// @brief Returns the geometry of boundary edge e.
        __device__ Edge boundary_edge(int e) const { return edge(boundary_edges[e]); }

    private:
        friend class Mesh2D;

        VectorWrapper<const double2> nodes; ///< (n_nodes,)
        const_imat_wrapper elems;           ///< (4, n_elem)
        const_imat_wrapper edge_nodes;      ///< (2, n_edges) — node indices per edge (pre-ordered for outward normal)
        const_ivec_wrapper interior_edges;  ///< indices into edge_nodes
        const_ivec_wrapper boundary_edges;  ///< indices into edge_nodes
    };

} // namespace cuddh

#endif
