#pragma once

#include <vector>

#include "HostDeviceArray.hpp"
#include "Mesh3D/Connectivity.hpp"
#include "Mesh3D/Element.hpp"
#include "Mesh3D/Face.hpp"
#include "QuadratureRule.hpp"
#include "SmallMatrix.hpp"
#include "Tensor.hpp"

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
        static Mesh3D uniform_cube(int nx, double ax, double bx, int ny, double ay, double by, int nz, double az,
                                   double bz);

        Mesh3D() = default;
        ~Mesh3D() = default;

        Mesh3D(const Mesh3D &) = delete;
        Mesh3D &operator=(const Mesh3D &) = delete;

        Mesh3D(Mesh3D &&) = default;
        Mesh3D &operator=(Mesh3D &&) = default;

        int n_elem() const { return nel; }
        int n_faces() const { return nf; }
        int n_boundary_faces() const { return nbf; }
        int n_interior_faces() const { return nif; }

        double h() const { return _h; }

        /**
         * @brief Returns the geometry of element el.
         */
        HexElement element(int el) const
        {
            cuddh_assert(0 <= el && el < n_elem(),
                         printf("Mesh3D error: element index %d out of range [0, %d).\n", el, n_elem()););

            auto elems = reshape(this->elems.host_read(), 8, nel);
            auto nodes = reshape(this->nodes.host_read(), this->nodes.size());

            double3 x[8];
            for (int i = 0; i < 8; ++i)
                x[i] = nodes[elems(i, el)];

            return HexElement(x);
        }

        /**
         * @brief Returns the geometry of face f.
         */
        QuadFace face(int f) const
        {
            cuddh_assert(0 <= f && f < n_faces(),
                         printf("Mesh3D error: face index %d out of range [0, %d).\n", f, n_faces()););

            auto faces = reshape(this->faces.host_read(), 4, nf);
            auto nodes = reshape(this->nodes.host_read(), this->nodes.size());

            double3 x[4];
            for (int i = 0; i < 4; ++i)
                x[i] = nodes[faces(i, f)];

            return QuadFace(x);
        }

        /**
         * @brief Returns the geometry of boundary face f.
         */
        QuadFace boundary_face(int f) const
        {
            cuddh_assert(
                0 <= f && f < n_boundary_faces(),
                printf("Mesh3D error: boundary face index %d out of range [0, %d).\n", f, n_boundary_faces()););

            const int *boundary_faces = this->boundary_faces.host_read();
            return face(boundary_faces[f]);
        }

        /**
         * @brief Returns the geometry of interior face f.
         */
        QuadFace interior_face(int f) const
        {
            cuddh_assert(
                0 <= f && f < n_interior_faces(),
                printf("Mesh3D error: interior face index %d out of range [0, %d).\n", f, n_interior_faces()););

            const int *interior_faces = this->interior_faces.host_read();
            return face(interior_faces[f]);
        }

        /**
         * @brief Returns the face connectivity of face f.
         */
        FaceConnectivity face_connectivity(int f) const
        {
            cuddh_assert(0 <= f && f < n_faces(),
                         printf("Mesh3D error: face index %d out of range [0, %d).\n", f, n_faces()););

            return connectivity[f];
        }

        /**
         * @brief Returns the face connectivity of the interior face f.
         */
        FaceConnectivity interior_face_connectivity(int f) const
        {
            cuddh_assert(
                0 <= f && f < n_interior_faces(),
                printf("Mesh3D error: interior face index %d out of range [0, %d).\n", f, n_interior_faces()););

            const int *interior_faces = this->interior_faces.host_read();
            return face_connectivity(interior_faces[f]);
        }

        /**
         * @brief Returns the face connectivity of the boundary face f.
         */
        FaceConnectivity boundary_face_connectivity(int f) const
        {
            cuddh_assert(
                0 <= f && f < n_boundary_faces(),
                printf("Mesh3D error: boundary face index %d out of range [0, %d).\n", f, n_boundary_faces()););

            const int *boundary_faces = this->boundary_faces.host_read();
            return face_connectivity(boundary_faces[f]);
        }

        /**
         * @brief Returns a list of face indices corresponding to the boundary faces.
         */
        const_ivec_wrapper get_boundary_faces() const { return reshape(boundary_faces.host_read(), nbf); }

        // Get the device mesh
        DeviceMesh3D to_device() const;

    private:
        int nel;   // number of elements
        int nf;    // number of faces
        int nbf;   // number of boundary faces
        int nif;   // number of interior faces
        double _h; // mesh size
        HostDeviceArray<double3> nodes;
        HostDeviceArray<int> elems;          // shape (8, n_elems) in canonical order
        HostDeviceArray<int> faces;          // shape (4, n_faces) -> indices of faces
        HostDeviceArray<int> interior_faces; // shape (n_interior_faces) -> indices of interior faces (in faces)
        HostDeviceArray<int> boundary_faces; // shape (n_boundary_faces) -> indices of boundary faces (in faces)
        HostDeviceArray<FaceConnectivity> _connectivity; // shape (n_faces) -> connectivity for each face

        std::vector<FaceConnectivity> connectivity;
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

        __host__ __device__ int n_elem() const { return elems.shape(1); }
        __host__ __device__ int n_faces() const { return faces.shape(1); }
        __host__ __device__ int n_boundary_faces() const { return boundary_faces.size(); }
        __host__ __device__ int n_interior_faces() const { return interior_faces.size(); }

        __device__ HexElement element(int el) const
        {
            cuddh_assert(0 <= el && el < n_elem(),
                         printf("Mesh3D error: element index %d out of range [0, %d).\n", el, n_elem()););

            double3 x[8];
            for (int i = 0; i < 8; ++i)
                x[i] = nodes[elems(i, el)];

            return HexElement(x);
        }

        __device__ QuadFace face(int f) const
        {
            cuddh_assert(0 <= f && f < n_faces(),
                         printf("Mesh3D error: face index %d out of range [0, %d).\n", f, n_faces()););

            double3 x[4];
            for (int i = 0; i < 4; ++i)
                x[i] = nodes[faces(i, f)];

            return QuadFace(x);
        }

        __device__ QuadFace boundary_face(int f) const
        {
            cuddh_assert(
                0 <= f && f < n_boundary_faces(),
                printf("Mesh3D error: boundary face index %d out of range [0, %d).\n", f, n_boundary_faces()););
            return face(boundary_faces[f]);
        }

        __device__ QuadFace interior_face(int f) const
        {
            cuddh_assert(
                0 <= f && f < n_interior_faces(),
                printf("Mesh3D error: interior face index %d out of range [0, %d).\n", f, n_interior_faces()););
            return face(interior_faces[f]);
        }

        /**
         * @brief Returns the face connectivity of face f.
         */
        __device__ FaceConnectivity face_connectivity(int f) const
        {
            cuddh_assert(0 <= f && f < n_faces(),
                         printf("Mesh3D error: face index %d out of range [0, %d).\n", f, n_faces()););
            return connectivity[f];
        }

    private:
        friend class Mesh3D;

        VectorWrapper<const double3> nodes;
        const_imat_wrapper elems;
        const_imat_wrapper faces;
        const_ivec_wrapper interior_faces;
        const_ivec_wrapper boundary_faces;
        VectorWrapper<const FaceConnectivity> connectivity;
    };
} // namespace cuddh
