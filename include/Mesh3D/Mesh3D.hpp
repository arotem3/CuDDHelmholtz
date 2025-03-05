#ifndef CUDDH_MESH_3D_HPP
#define CUDDH_MESH_3D_HPP

#include <unordered_map>
#include <array>
#include <algorithm>

#include "Tensor.hpp"
#include "QuadratureRule.hpp"
#include "SmallMatrix.hpp"
#include "Mesh3D/Element.hpp"
#include "Mesh3D/Face.hpp"
#include "Mesh3D/Connectivity.hpp"

#include "HostDeviceArray.hpp"

namespace cuddh
{
    class DeviceMesh3D;

    /**
     * @brief A 3D mesh of hexahedral elements.
     */
    class Mesh3D
    {
    public:
        /**
         * @brief construct a mesh of HexElements from a list of vertices
         * and a list of indices indicating the vertices of each element.
         *
         * The vertices of an element are ordered such that the first four
         * vertices are the corners of the bottom face in counter-clockwise
         * order, and the last four vertices are the corners of the top face
         * in counter-clockwise order.
         *
         * @param nx number of vertices
         * @param nodes The coordinates of the vertices
         * @param nel number of elements
         * @param elems shape (8, nel). The element corners. if j = elems(i, el),
         * then the i-th corner of element el is at node j.
         *
         * @return Mesh3D
         */
        static Mesh3D from_vertices(int nx, const double3 *nodes, int nel, const int *elems);

        /**
         * @brief Constructs a uniform structured mesh for the
         * cube [ax, bx] x [ay, by] x [az, bz] with nx * ny * nz elements.
         */
        static Mesh3D uniform_cube(int nx, double ax, double bx, int ny, double ay, double by, int nz, double az, double bz);

        Mesh3D() = default;
        ~Mesh3D() = default;

        Mesh3D(const Mesh3D &) = delete;
        Mesh3D &operator=(const Mesh3D &) = delete;

        Mesh3D(Mesh3D &&) = default;
        Mesh3D &operator=(Mesh3D &&) = default;

        int n_elem() const { return nel; }
        int n_boundary_faces() const { return nbf; }
        int n_interior_faces() const { return nif; }

        // Geometry of element el
        HexElement element(int el) const
        {
            if (el < 0 || el >= n_elem())
                throw std::out_of_range("Mesh3D::element: element index out of range.");

            auto elems = reshape(this->elems.host_read(), 8, nel);
            auto nodes = reshape(this->nodes.host_read(), this->nodes.size());

            double3 x[8];
            for (int i = 0; i < 8; ++i)
                x[i] = nodes[elems(i, el)];

            return HexElement(x);
        }

        // Geometry of boundary face f
        QuadFace boundary_face(int f) const
        {
            if (f < 0 || f >= n_boundary_faces())
                throw std::out_of_range("Mesh3D::boundary_faces: face index out of range.");

            auto boundary_faces = reshape(this->boundary_faces.host_read(), 4, nbf);
            auto nodes = reshape(this->nodes.host_read(), this->nodes.size());

            double3 x[4];
            for (int i = 0; i < 4; ++i)
                x[i] = nodes[boundary_faces(i, f)];

            return QuadFace(x);
        }

        // Geometry of interior face f
        QuadFace interior_face(int f) const
        {
            if (f < 0 || f >= n_interior_faces())
                throw std::out_of_range("Mesh3D::interior_faces: face index out of range.");

            auto interior_faces = reshape(this->interior_faces.host_read(), 4, nif);
            auto nodes = reshape(this->nodes.host_read(), this->nodes.size());

            double3 x[4];
            for (int i = 0; i < 4; ++i)
                x[i] = nodes[interior_faces(i, f)];

            return QuadFace(x);
        }

        // Face connectivity of interior face f
        FaceConnectivity interior_face_connectivity(int f) const
        {
            if (f < 0 || f >= n_interior_faces())
                throw std::out_of_range("Mesh3D::interior_face_connectivity: face index out of range.");

            return interior_connectivity[f];
        }

        // Face connectivity of boundary face f
        FaceConnectivity boundary_face_connectivity(int f) const
        {
            if (f < 0 || f >= n_boundary_faces())
                throw std::out_of_range("Mesh3D::boundary_face_connectivity: face index out of range.");

            return boundary_connectivity[f];
        }

        // Get the device mesh
        DeviceMesh3D to_device() const;

    private:
        int nel; // number of elements
        int nbf; // number of boundary faces
        int nif; // number of interior faces
        HostDeviceArray<double3> nodes;
        HostDeviceArray<int> elems;          // shape (8, n_elems) in canonical order
        HostDeviceArray<int> interior_faces; // shape (4, n_interior_faces) -> indices of interior faces
        HostDeviceArray<int> boundary_faces; // shape (4, n_boundary_faces) -> indices of boundary faces

        std::vector<FaceConnectivity> interior_connectivity;
        std::vector<FaceConnectivity> boundary_connectivity;
    };

    /**
     * @brief A 3D mesh of hexahedral elements on the device. Contains only geometric information (no connectivity).
     */
    class DeviceMesh3D
    {
    public:
        DeviceMesh3D() = default;
        DeviceMesh3D(const DeviceMesh3D &) = default;
        DeviceMesh3D(DeviceMesh3D &&) = default;
        DeviceMesh3D &operator=(const DeviceMesh3D &) = default;
        DeviceMesh3D &operator=(DeviceMesh3D &&) = default;

        int n_elem() const { return elems.shape(1); }
        int n_boundary_faces() const { return boundary_faces.shape(1); }
        int n_interior_faces() const { return interior_faces.shape(1); }

        __device__ HexElement element(int el) const
        {
#ifdef CUDDH_DEBUG
            if (el < 0 || el >= n_elem())
                cuddh_error("DeviceMesh3D::element: element index out of range.");
#endif

            double3 x[8];
            for (int i = 0; i < 8; ++i)
                x[i] = nodes[elems(i, el)];

            return HexElement(x);
        }

        __device__ QuadFace boundary_face(int f) const
        {
#ifdef CUDDH_DEBUG
            if (f < 0 || f >= n_boundary_faces())
                cuddh_error("DeviceMesh3D::boundary_faces: face index out of range.");
#endif

            double3 x[4];
            for (int i = 0; i < 4; ++i)
                x[i] = nodes[boundary_faces(i, f)];

            return QuadFace(x);
        }

        __device__ QuadFace interior_face(int f) const
        {
#ifdef CUDDH_DEBUG
            if (f < 0 || f >= n_interior_faces())
                cuddh_error("DeviceMesh3D::interior_faces: face index out of range.");
#endif

            double3 x[4];
            for (int i = 0; i < 4; ++i)
                x[i] = nodes[interior_faces(i, f)];

            return QuadFace(x);
        }

    private:
        friend class Mesh3D;

        VectorWrapper<const double3> nodes;
        const_imat_wrapper elems;
        const_imat_wrapper interior_faces;
        const_imat_wrapper boundary_faces;
    };
} // namespace cuddh

#endif
