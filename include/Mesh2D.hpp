#ifndef CUDDH_MESH_2D_HPP
#define CUDDH_MESH_2D_HPP

#include <unordered_map>

#include "Tensor.hpp"
#include "Node.hpp"
#include "Edge.hpp"
#include "Element.hpp"
#include "QuadratureRule.hpp"

namespace cuddh
{
    /// @brief The 2D mesh.
    class Mesh2D
    {
    public:
        /// @brief manages arrays of element metric information
        class ElementMetricCollection
        {
        public:
            /// @brief initialize collection of metrics on elements
            /// @param mesh the mesh
            /// @param quad 1D quadrature rule to evaluate metrics on
            ElementMetricCollection(const Mesh2D& mesh, const QuadratureRule& quad);

            ElementMetricCollection(ElementMetricCollection&&);

            /// returns an array of the element jacobians evaluated on a quadrature rule.
            /// The output J has shape (2, 2, n, n, n_elem).
            const double * jacobians() const;
            
            /// returns an array of the element measures ie the determinant of the
            /// jacobians evaluated on a quadrature rule. The output detJ has shape (n, n, n_elem).
            const double * measures() const;
            
            /// returns an array of the physical coordinates of the quadrature rule on
            /// every element. The output x has shape (2, n, n, n_elem).
            const double * physical_coordinates() const;

        private:
            const Mesh2D& mesh;
            const QuadratureRule& quad;

            mutable std::unique_ptr<double[]> J;
            mutable std::unique_ptr<double[]> detJ;
            mutable std::unique_ptr<double[]> x;
        };

        /// @brief manages arrays of element metric information
        class EdgeMetricCollection
        {
        public:
            /// @brief initialize collection of metrics on edges
            /// @param mesh the 2d mesh
            /// @param edge_type interior or boundary
            /// @param quad the quadrature rule to evaluate metrics on
            EdgeMetricCollection(const Mesh2D& mesh, const FaceType edge_type, const QuadratureRule& quad);

            EdgeMetricCollection(const Mesh2D& mesh, int n_faces, const int * faces, const QuadratureRule& quad);

            EdgeMetricCollection(EdgeMetricCollection&&);

            /// return an array of the edge measures on the quadrature rule for edges of
            /// the requested types. The output has shape (n, n_edges).
            const double * measures() const;

            /// returns an array of the physical coordinates of the quadrature rule on
            /// every edge of the requested type. The output has shape (2, n, n_edges)
            const double * physical_coordinates() const;

            /// returns an array of the normal derivatives of all of the edges of the
            /// requested FaceType evaluated on the quadrature rule. The output has shape
            /// (2, n, n_edges).
            const double * normals() const;

        private:
            const Mesh2D& mesh;
            const QuadratureRule& quad;
            const FaceType edge_type;

            const bool face_subset;
            const_ivec_wrapper _faces;

            mutable std::unique_ptr<double[]> detJ;
            mutable std::unique_ptr<double[]> x;
            mutable std::unique_ptr<double[]> n;
        };

    private:
        std::vector<Node> _nodes;
        std::vector<std::unique_ptr<Edge>> _edges;
        std::vector<std::unique_ptr<Element>> _elements;
        std::vector<int> _interior_nodes;
        std::vector<int> _boundary_nodes;
        std::vector<int> _boundary_edges;
        std::vector<int> _interior_edges;

        mutable std::unordered_map<std::string, ElementMetricCollection> elem_collections;
        mutable std::unordered_map<std::string, EdgeMetricCollection> interior_edge_collections;
        mutable std::unordered_map<std::string, EdgeMetricCollection> boundary_edge_collections;

    public:
        /// @brief constructs empty mesh
        Mesh2D() {}
        ~Mesh2D() = default;

        /// @brief DELETED: mesh maintains unique pointers to abstract types. Copies are non-trivial.
        Mesh2D(const Mesh2D &mesh) = delete;

        /// @brief move mesh
        Mesh2D(Mesh2D &&) = default;

        /// @brief DELETED: mesh maintains unique pointers to abstract types. Copies are non-trivial. 
        Mesh2D &operator=(const Mesh2D &) = delete;

        /// @brief move mesh 
        Mesh2D &operator=(Mesh2D &&) = default;

        /// number of elements in mesh. If mesh is distributed (MPI), then
        /// returns the number of elements on this processor.
        int n_elem() const
        {
            return _elements.size();
        }

        /// @brief return number of edges in mesh. 
        int n_edges() const
        {
            return _edges.size();
        }

        /// number of edges of specified type in mesh.
        int n_edges(FaceType type) const
        {
            if (type == FaceType::BOUNDARY)
            {
                return _boundary_edges.size();
            }
            else
            {
                return _interior_edges.size();
            }
        }

        /// @brief  return the number of nodes in mesh.
        int n_nodes() const
        {
            return _nodes.size();
        }

        /// @brief number of node of specified type in mesh. 
        int n_nodes(NodeType type) const
        {
            if (type == NodeType::BOUNDARY)
                return _boundary_nodes.size();
            else
                return _interior_nodes.size();
        }

        /// @brief returns the maximum polynomial degree of all element mappings.
        int max_element_order() const
        {
            // only bilinear elements supported so far
            return 1;
        }

        /// @brief returns the minimum polynomial degree of all element mappings. 
        int min_element_order() const
        {
            // only bilinear elements supported so far
            return 1;
        }

        /// @brief returns the shortest length scale (edge length) of the mesh.
        double min_h() const;

        /// @brief returns the longest length scale (edge length) of the mesh.
        double max_h() const;

        const Node& node(int i) const
        {
        #ifdef CUDDH_DEBUG
            if (i < 0 || i >= (int)_nodes.size())
                cuddh_error("Mesh2D::node error: node index out of range.");
        #endif

            return _nodes[i];
        }

        const Node& node(int i, NodeType type) const
        {
            if (type == NodeType::BOUNDARY)
            {
            #ifdef CUDDH_DEBUG
                if (i < 0 || i >= (int)_boundary_nodes.size())
                    cuddh_error("Mesh2D::node error: boundary node index out of range.");
            #endif

                return _nodes[_boundary_nodes[i]];
            }
            else
            {
            #ifdef CUDDH_DEBUG
                if (i < 0 || i >= (int)_interior_nodes.size())
                    cuddh_error("Mesh2D::node error: interior node index out of range.");
            #endif
            
                return _nodes[_interior_nodes[i]];
            }
        }

        const Edge *edge(int i) const
        {
        #ifdef CUDDH_DEBUG
            if (i < 0 || i >= (int)_edges.size())
                cuddh_error("Mesh2D::edge error: edge index out of range.");
        #endif

            return _edges[i].get();
        }

        /// returns the edge of FaceType type specified by edge index i.
        const Edge *edge(int i, FaceType type) const
        {
            if (type == FaceType::BOUNDARY)
            {
            #ifdef CUDDH_DEBUG
                if (i < 0 || i >= (int)_boundary_edges.size())
                    cuddh_error("Mesh2D::edge error: boundary edge index out of range.");
            #endif

                return _edges[_boundary_edges[i]].get();
            }
            else
            {
            #ifdef CUDDH_DEBUG
                if (i < 0 || i >= (int)_interior_edges.size())
                    cuddh_error("Mesh2D::edge error: interior edge index out of range.");
            #endif

                return _edges[_interior_edges[i]].get();
            }
        }

        /// @brief returns the indicies of the edges such that edge(i) is on the boundary
        ivec boundary_edges() const;

        /// returns the element specified by element index el. For distributed
        /// meshes: this index is local to the processor and should be in the
        /// range [0, n_elem() ).
        const Element *element(int el) const
        {
        #ifdef CUDDH_DEBUG
            if (el < 0 || el >= (int)_elements.size())
                cuddh_error("Mesh2D::element error: element index out of range.");
        #endif

            return _elements[el].get();
        }

        const ElementMetricCollection& element_metrics(const QuadratureRule& quad) const;

        const EdgeMetricCollection& edge_metrics(const QuadratureRule& quad, FaceType edge_type) const;

        /// @brief constructs a mesh of QuadElements given a list vertices x and a
        /// list of indices indicating the vertices of each element.
        /// @param[in] nx number of vertices
        /// @param[in] x shape (2, nx). The coordinates of the vertices
        /// @param[in] nel number of elements
        /// @param[in] elems shape (4, nel). The element corners. if j = elems(i, el)
        /// then the i-th corner of element el is x(*, j).
        /// @return mesh
        static Mesh2D from_vertices(int nx, const double *x, int nel, const int *elems);

        /// @brief construct a uniform structured mesh for the rectangle [ @a ax , @a bx ] x [ @a ay , @a by ] with @a nx by @a ny elements.
        /// @param nx number of elements to partition [ @a ax , @a bx ]
        /// @param ax lower bound for x
        /// @param bx upper bound for x
        /// @param ny number of elements to partition [ @a ay , @a by ]
        /// @param ay lower bound for y
        /// @param by upper bound for y
        /// @return the mesh.
        static Mesh2D uniform_rect(int nx, double ax, double bx, int ny, double ay, double by);

    private:
        // resets all data
        inline void reset()
        {
            _edges.clear();
            _elements.clear();
            _boundary_edges.clear();
            _interior_edges.clear();
            elem_collections.clear();
            interior_edge_collections.clear();
            boundary_edge_collections.clear();
        }
    };

    inline Mesh2D::ElementMetricCollection::ElementMetricCollection(const Mesh2D& mesh_, const QuadratureRule& quad_) : mesh(mesh_), quad{quad_} {}

    inline Mesh2D::ElementMetricCollection::ElementMetricCollection(ElementMetricCollection&& a) : mesh(a.mesh), quad{a.quad}, J(std::move(a.J)), detJ(std::move(a.detJ)), x(std::move(a.x)) {}

    inline Mesh2D::EdgeMetricCollection::EdgeMetricCollection(const Mesh2D& mesh_, const FaceType edge_type_, const QuadratureRule& quad_) : mesh(mesh_), quad{quad_}, edge_type(edge_type_), face_subset{false}, _faces() {}

    inline Mesh2D::EdgeMetricCollection::EdgeMetricCollection(EdgeMetricCollection&& a) : mesh(a.mesh), quad{a.quad}, edge_type(a.edge_type), face_subset{a.face_subset}, _faces(std::move(a._faces)), detJ(std::move(a.detJ)), x(std::move(a.x)), n(std::move(a.n)) {}

    inline Mesh2D::EdgeMetricCollection::EdgeMetricCollection(const Mesh2D& mesh_, int n_faces, const int * faces, const QuadratureRule& quad_) : mesh{mesh_}, quad{quad_}, edge_type{FaceType::INTERIOR}, face_subset{true}, _faces(faces, n_faces) {}

} // namespace cuddh


#endif