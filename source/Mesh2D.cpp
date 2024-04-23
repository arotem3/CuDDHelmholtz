#include "Mesh2D.hpp"

template <typename Map, typename Key>
static bool contains(const Map & map, Key key)
{
    return map.find(key) != map.end();
}

namespace cuddh
{
    Mesh2D Mesh2D::from_vertices(int nx, const double * x_, int nel, const int * elems_)
    {
        auto coo = reshape(x_, 2, nx);
        auto elems = reshape(elems_, 4, nel);

        constexpr int emap1[] = {0,1,3,0};
        constexpr int emap2[] = {1,2,2,3};

        Mesh2D mesh;
        mesh._elements.resize(nel);
        mesh._nodes.resize(nx);

        // construct nodes
        for (int k = 0; k < nx; ++k)
        {
            auto& node = mesh._nodes[k];
            node.x[0] = coo(0, k);
            node.x[1] = coo(1, k);
            node.type = NodeType::BOUNDARY;
            node.id = k;
        }
        
        dmat x(2, 4);
        int cs[4];

        // construct elements
        for (int el = 0; el < nel; ++el)
        {
            for (int i = 0; i < 4; ++i)
            {
                const int c = elems(i, el);
                cs[i] = c;
                x(0, i) = coo(0, c);
                x(1, i) = coo(1, c);

                auto& node = mesh._nodes[c];
                
                Node::element_info info;
                info.i = i;
                info.id = el;
                
                node.connected_elements.push_back(info);
            }

            auto& elem = mesh._elements[el];
            elem.reset(new QuadElement(x.data()));
            elem->id = el;
            for (int i = 0; i < 4; ++i)
                elem->nodes[i] = cs[i];
        }

        // construct edges
        std::unordered_map<int, int> edge_map;
        auto key = [nx](int i, int j) -> int
        {
            return std::min(i, j) + nx * std::max(i, j);
        };
        int edge_id = 0;

        for (int el = 0; el < nel; ++el)
        {
            for (int s = 0; s < 4; ++s)
            {
                const int l1 = emap1[s];
                const int l2 = emap2[s];

                const int C0 = elems(l1, el);
                const int C1 = elems(l2, el);

                const int k = key(C0, C1);
                if (not contains(edge_map, k))
                {
                    const double * x0 = x_ + 2*C0;
                    const double * x1 = x_ + 2*C1;
                    mesh._edges.push_back(std::unique_ptr<Edge>(new StraightEdge(x0, x1, s)));
                    Edge * edge = mesh._edges[edge_id].get();

                    edge->nodes[0] = C0;
                    edge->nodes[1] = C1;
                    edge->elements[0] = el;
                    edge->id = edge_id;
                    edge->sides[0] = s;
                    edge->type = FaceType::BOUNDARY;
                    edge->delta = 1;
                    edge_map[k] = edge_id++;
                }
                else
                {
                    int e = edge_map.at(k);
                    Edge * edge = mesh._edges[e].get();

                    int e0 = edge->elements[0];
                    int s0 = edge->sides[0];
                    int n1 = elems(emap1[s0], e0);

                    edge->elements[1] = el;
                    edge->sides[1] = s;
                    edge->type = FaceType::INTERIOR;
                    edge->delta = (C0 == n1) ? 1 : -1;

                    mesh._nodes[C0].type = NodeType::INTERIOR;
                    mesh._nodes[C1].type = NodeType::INTERIOR;
                }
            }
        }

        // boundary edges
        for (const auto& edge : mesh._edges)
        {
            if (edge->type == FaceType::BOUNDARY)
                mesh._boundary_edges.push_back(edge->id);
            else
                mesh._interior_edges.push_back(edge->id);
        }

        // boundary nodes
        for (const auto& node : mesh._nodes)
        {
            if (node.type == NodeType::BOUNDARY)
                mesh._boundary_nodes.push_back(node.id);
            else
                mesh._interior_nodes.push_back(node.id);
        }

        return mesh;
    }

    Mesh2D Mesh2D::uniform_rect(int nx, double ax, double bx, int ny, double ay, double by)
    {
        int np = (nx+1)*(ny+1);
        int nel = nx*ny;
        dcube coo(2, nx+1, ny+1);
        Cube<int> elems(4, nx, ny);

        auto l = [nx,ny](int i, int j) -> int {return i + (nx+1)*j;};

        double dx = (bx - ax) / nx;
        double dy = (by - ay) / ny;
        for (int j=0; j <= ny; ++j)
        {
            const double y = ay + dy * j;
            for (int i=0; i <= nx; ++i)
            {
                coo(0, i, j) = ax + dx * i;
                coo(1, i, j) = y;
            }
        }
        
        for (int j=0; j < ny; ++j)
        {
            for (int i=0; i < nx; ++i)
            {
                elems(0, i, j) = l(  i,   j);
                elems(1, i, j) = l(i+1,   j);
                elems(2, i, j) = l(i+1, j+1);
                elems(3, i, j) = l(  i, j+1);
            }
        }

        return from_vertices(np, coo, nel, elems);
    }

    template <typename EvalMetric>
    static void set_element_metric(host_device_dvec& metric_, int dim, const Mesh2D& mesh, const QuadratureRule& quad, EvalMetric eval_metric)
    {
        const int m = quad.size();
        const int nel = mesh.n_elem();

        metric_.resize(dim * m * m * nel);
        auto metric = reshape(metric_.host_write(), dim, m, m, nel);

        double xi[2];

        for (int el=0; el < nel; ++el)
        {
            const Element * elem = mesh.element(el);
            for (int j = 0; j < m; ++j)
            {
                xi[1] = quad.x(j);
                for (int i = 0; i < m; ++i)
                {
                    xi[0] = quad.x(i);
                    eval_metric(&metric(0, i, j, el), elem, xi);
                }
            }
        }
    }

    const double * Mesh2D::ElementMetricCollection::jacobians(MemorySpace m) const
    {
        if (J.size() == 0)
        {
            set_element_metric(J, 4, mesh, quad, [](double* metric, const Element * elem, const double * xi) -> void {elem->jacobian(xi, metric);});
        }

        return J.read(m);
    }

    const double * Mesh2D::ElementMetricCollection::measures(MemorySpace m) const
    {
        if (detJ.size() == 0)
        {
            set_element_metric(detJ, 1, mesh, quad, [](double* metric, const Element * elem, const double * xi) -> void {*metric = elem->measure(xi);});
        }

        return detJ.read(m);
    }

    const double * Mesh2D::ElementMetricCollection::physical_coordinates(MemorySpace m) const
    {
        if (x.size() == 0)
        {
            set_element_metric(x, 2, mesh, quad, [](double* metric, const Element * elem, const double * xi) -> void {elem->physical_coordinates(xi, metric);});
        }

        return x.read(m);
    }

    template <typename EvalMetric>
    static void set_edge_metric(host_device_dvec& metric_, int dim, FaceType edge_type, const Mesh2D& mesh, const QuadratureRule& quad, EvalMetric eval_metric)
    {
        const int m = quad.size();
        const int ne = mesh.n_edges(edge_type);

        metric_.resize(dim * m * ne);
        auto metric = reshape(metric_.host_write(), dim, m, ne);

        for (int e = 0; e < ne; ++e)
        {
            const Edge * E = mesh.edge(e, edge_type);

            for (int i = 0; i < m; ++i)
            {
                double xi = quad.x(i);
                
                eval_metric(&metric(0, i, e), E, xi);
            }
        }
    }

    template <typename EvalMetric>
    static void set_edge_metric(host_device_dvec& metric_, int dim, const_ivec_wrapper faces, const Mesh2D& mesh, const QuadratureRule& quad, EvalMetric eval_metric)
    {
        const int m = quad.size();
        const int ne = faces.size();

        metric_.resize(dim * m * ne);
        auto metric = reshape(metric_.host_write(), dim, m, ne);

        for (int e = 0; e < ne; ++e)
        {
            const Edge * E = mesh.edge(faces(e));

            for (int i = 0; i < m; ++i)
            {
                const double xi = quad.x(i);
                eval_metric(&metric(0, i, e), E, xi);
            }
        }
    }

    const double * Mesh2D::EdgeMetricCollection::measures(MemorySpace m) const
    {
        if (detJ.size() == 0)
        {
            auto eval = [](double* metric, const Edge * E, double xi) -> void {*metric = E->measure(xi);};
            if (face_subset)
                set_edge_metric(detJ, 1, _faces, mesh, quad, eval);
            else
                set_edge_metric(detJ, 1, edge_type, mesh, quad, eval);
        }

        return detJ.read(m);
    }

    const double * Mesh2D::EdgeMetricCollection::physical_coordinates(MemorySpace m) const
    {
        if (x.size() == 0)
        {
            auto eval = [](double * metric, const Edge * E, double xi) -> void {E->physical_coordinates(xi, metric);};
            if (face_subset)
                set_edge_metric(x, 2, _faces, mesh, quad, eval);
            else
                set_edge_metric(x, 2, edge_type, mesh, quad, eval);
        }

        return x.read(m);
    }

    const double * Mesh2D::EdgeMetricCollection::normals(MemorySpace m) const
    {
        if (n.size() == 0)
        {
            auto eval = [](double * metric, const Edge * E, double xi) -> void {E->normal(xi, metric);};
            if (face_subset)
                set_edge_metric(n, 2, _faces, mesh, quad, eval);
            else
                set_edge_metric(n, 2, edge_type, mesh, quad, eval);
        }

        return n.read(m);
    }

    double Mesh2D::min_h() const
    {
        double h = std::numeric_limits<double>::infinity();
        for (auto& edge : _edges)
        {
            h = std::min(h, edge->length());
        }
        return h;
    }

    double Mesh2D::max_h() const
    {
        double h = -1;
        for (auto& edge : _edges)
        {
            h = std::max(h, edge->length());
        }
        return h;
    }

    ivec Mesh2D::boundary_edges() const
    {
        const int ne = _boundary_edges.size();
        ivec b(ne);
        for (int i = 0; i < ne; ++i)
            b(i) = _boundary_edges.at(i);
        return b;
    }

    const Mesh2D::ElementMetricCollection& Mesh2D::element_metrics(const QuadratureRule& quad) const
    {
        auto id = quad.name();
        if (not contains(elem_collections, id))
        {
            elem_collections.insert({id, ElementMetricCollection(*this, quad)});
        }

        return elem_collections.at(id);
    }

    const Mesh2D::EdgeMetricCollection& Mesh2D::edge_metrics(const QuadratureRule& quad, FaceType edge_type) const
    {
        auto& collection = (edge_type == FaceType::INTERIOR) ? interior_edge_collections : boundary_edge_collections;

        auto id = quad.name();
        if (not contains(collection, id))
        {
            collection.insert({id, EdgeMetricCollection(*this, edge_type, quad)});
        }

        return collection.at(id);
    }
} // namespace cuddh
