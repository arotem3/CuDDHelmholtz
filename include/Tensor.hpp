#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <vector>
#include <stdexcept>
#include <memory>

#include "cuddh_error.hpp"

namespace cuddh
{
    template <typename Size, typename... Sizes>
    inline int tensor_dims(int * dim, Size s, Sizes... shape)
    {
        if (s < 0)
            cuddh_error("Tensor error: tensor cannot have negative dimensions.");
        
        *dim = s;
        if constexpr (sizeof...(shape) > 0)
        {
            ++dim;
            return s * tensor_dims(dim, shape...);
        }
        else
            return s;
    }

    template <typename Ind, typename... Inds>
    inline int tensor_index(const int * shape, Ind idx, Inds... ids)
    {
        #ifdef CUDDH_DEBUG
        if (idx < 0 || idx >= *shape)
            cuddh_error("Tensor error: tensor index out of range.");
        #endif

        if constexpr (sizeof...(ids) > 0)
        {
            const int dim = *shape;
            ++shape;
            return idx + dim * tensor_index(shape, ids...);
        }
        else
            return idx;
    }

    /// @brief provides read/write access to an externally managed array with
    /// high dimensional indexing.
    /// @tparam scalar the type of array, e.g. double, int, etc.
    /// @tparam Dim the tensor dimension, e.g. 2 for a matrix
    template <int Dim, typename scalar>
    class TensorWrapper
    {
    protected:
        int _shape[Dim];
        int len;
        scalar * ptr;
    
    public:
        /// @brief empty tensor
        TensorWrapper() : _shape{0}, len{0}, ptr(nullptr) {};

        virtual ~TensorWrapper() = default;
        
        /// @brief copy tensor
        /// @param[in] tensor to copy
        TensorWrapper(const TensorWrapper&) = default;

        /// @brief copy tensor
        /// @param[in] tensor to copy
        /// @return `this`
        TensorWrapper& operator=(const TensorWrapper&) = default;

        /// @brief wrap externally managed array
        /// @tparam ...Sizes sequence of `int`
        /// @param[in] data_ externally managed array
        /// @param[in] ...shape_ shape of array as a sequence of `int`s
        template <typename... Sizes>
        inline explicit TensorWrapper(scalar * data_, Sizes... shape_) : ptr(data_)
        {
            static_assert(Dim > 0, "Tensor must have a positive number of dimensions");
            static_assert(sizeof...(shape_) == Dim, "Wrong number of dimensions specified.");
            
            len = tensor_dims(_shape, shape_...);
        }

        /// @brief high dimensional read/write access.
        /// @tparam ...Indices sequence of `int`
        /// @param[in] ...ids indices
        /// @return reference to data at index (`...ids`)
        template <typename... Indices>
        inline scalar& at(Indices... ids)
        {
            static_assert(sizeof...(ids) == Dim, "Wrong number of indices specified.");

            #ifdef CUDDH_DEBUG
            if (ptr == nullptr)
                cuddh_error("TensorWrapper::at error: memory uninitialized.");
            #endif

            return ptr[tensor_index(_shape, ids...)];
        }

        /// @brief high dimensional read-only access.
        /// @tparam ...Indices sequence of `int`
        /// @param[in] ...ids indices
        /// @return const reference to data at index (`...ids`)
        template <typename... Indices>
        inline const scalar& at(Indices... ids) const
        {
            static_assert(sizeof...(ids) == Dim, "Wrong number of indices specified.");

            #ifdef CUDDH_DEBUG
            if (ptr == nullptr)
                cuddh_error("TensorWrapper::at error: memory uninitialized.");
            #endif

            return ptr[tensor_index(_shape, ids...)];
        }

        /// @brief high dimensional read/write access.
        /// @tparam ...Indices sequence of `int`
        /// @param[in] ...ids indices
        /// @return reference to data at index (`...ids`)
        template <typename... Indices>
        inline scalar& operator()(Indices... ids)
        {
            return at(std::forward<Indices>(ids)...);
        }

        /// @brief high dimensional read-only access.
        /// @tparam ...Indices sequence of `int`
        /// @param[in] ...ids indices
        /// @return const reference to data at index (`...ids`)
        template <typename... Indices>
        inline const scalar& operator()(Indices... ids) const
        {
            return at(std::forward<Indices>(ids)...);
        }

        /// @brief linear indexing. read/write access.
        /// @param[in] idx flattened index
        /// @return reference to data at linear index `idx`.
        inline scalar& operator[](int idx)
        {
            #ifdef CUDDH_DEBUG
            if (ptr == nullptr)
                cuddh_error("TensorWrapper::operator[] error: memory uninitialized.");
            if (idx < 0 || idx >= len)
                cuddh_error("TensorWrapper::operator[] error: linear index out of range.");
            #endif

            return ptr[idx];
        }

        /// @brief linear indexing. read only access.
        /// @param[in] idx flattened index
        /// @return const reference to data at linear index `idx`.
        inline const scalar& operator[](int idx) const
        {
            #ifdef CUDDH_DEBUG
            if (ptr == nullptr)
                cuddh_error("TensorWrapper::operator[] error: memory uninitialized.");
            if (idx < 0 || idx >= len)
                cuddh_error("TensorWrapper::operator[] error: linear index out of range.");
            #endif

            return ptr[idx];
        }
    
        /// @brief implicit conversion to scalar* where the returned pointer is
        /// the one managed by the tensor.
        inline operator scalar*()
        {
            return ptr;
        }

        /// @brief implicit conversion to scalar* where the returned pointer is
        /// the one managed by the tensor.
        inline operator const scalar*() const
        {
            return ptr;
        }

        /// @brief returns the externally managed array 
        inline scalar * data()
        {
            return ptr;
        }

        /// @brief returns read-only pointer to the externally managed array
        inline const scalar * data() const
        {
            return ptr;
        }
    
        inline scalar * begin()
        {
            return ptr;
        }

        inline scalar * end()
        {
            return ptr + len;
        }

        inline const scalar * begin() const
        {
            return ptr;
        }

        inline const scalar * end() const
        {
            return ptr + len;
        }

        /// @brief returns the shape of the tensor. Has length `Dim` 
        inline const int * shape() const
        {
            return _shape;
        }    

        inline int shape(int d) const
        {
            #ifdef CUDDH_DEBUG
            if (d < 0 || d >= Dim)
                cuddh_error("TensorWrapper::shape() error: shape index out of range of Dim.");
            #endif
            return _shape[d];
        }
    
        /// @brief returns total size of tensor. The product of shape.
        inline int size() const
        {
            return len;
        }
    
        inline void fill(scalar x) requires (!std::is_const_v<scalar>)
        {
            for (int i=0; i < len; ++i)
            {
                ptr[i] = x;
            }
        }

        inline void zeros() requires (!std::is_const_v<scalar>)
        {
            fill(0);
        }

        inline void ones() requires (!std::is_const_v<scalar>)
        {
            fill(1);
        }
    };

    /// @brief wraps an array in a `TensorWrapper`. Same as declaring a new
    /// `TensorWrapper<sizeof...(Sizes), scalar>(data, shape).`
    /// @tparam scalar type of array
    /// @tparam ...Sizes sequence of `int`
    /// @param[in] data the array
    /// @param[in] ...shape the shape of the tensor
    template <typename scalar, typename... Sizes>
    inline TensorWrapper<sizeof...(Sizes), scalar> reshape(scalar * data, Sizes... shape)
    {
        return TensorWrapper<sizeof...(Sizes), scalar>(data, shape...);
    }

    /// @brief reshape `TensorWrapper`. Returns new TensorWrapper with new shape
    /// but points to same data.
    /// @tparam scalar type of array
    /// @tparam ...Sizes sequence of `int`
    /// @param[in] tensor the array
    /// @param[in] ...shape the shape of the tensor
    template <typename scalar, int Dim, typename... Sizes>
    inline TensorWrapper<sizeof...(Sizes), scalar> reshape(TensorWrapper<Dim, scalar> tensor, Sizes... shape)
    {
        return reshape(tensor.data(), std::forward<Sizes>(shape)...);
    }

    /// @brief A `TensorWrapper` where the data is internally managed.
    /// @tparam scalar type of array. e.g. double, int
    /// @tparam Dim dimension of tensor. e.g. a matrix has `Dim == 2`
    template <int Dim, typename scalar>
    class Tensor : public TensorWrapper<Dim, scalar>
    {
    private:
        std::unique_ptr<scalar[]> mem;

    public:
        /// @brief empty tensor
        inline Tensor() : TensorWrapper<Dim, scalar>() {}

        ~Tensor() = default;
  
        /// @brief move tensor
        Tensor(Tensor&&) = default;

        /// @brief move tensor 
        Tensor& operator=(Tensor&&) = default;

        /// @brief copy tensor
        Tensor(const Tensor<Dim, scalar>&);

        /// @brief copy tensor
        Tensor& operator=(const Tensor&);

        /// @brief new tensor of specified shape initialized with default constructor (0 for numeric types).
        /// @tparam ...Sizes sequence of `int`s
        /// @param[in] ...sizes_ shape
        template <typename... Sizes>
        explicit Tensor(Sizes... shape_)
            : TensorWrapper<Dim, scalar>(nullptr, shape_...),
              mem(new scalar[this->len]())
        {
            this->ptr = mem.get();
        }

        /// @brief resizes the tensor, reallocating memory if more memory is
        /// needed. The data should be assumed to be unitialized.
        /// @tparam ...Sizes sequence of `int`s
        /// @param ...shape_ new shape
        template <typename... Sizes>
        inline void reshape(Sizes... shape_)
        {
            static_assert(sizeof...(shape_) == Dim, "Wrong number of dimensions specified.");
            
            int new_len = tensor_dims(this->_shape, shape_...);

            if (new_len > this->len)
            {
                mem.reset(new scalar[new_len]());
                this->ptr = mem.get();
            }
            this->len = new_len;
        }
    };

    template <int Dim, typename scalar>
    Tensor<Dim, scalar>::Tensor(const Tensor<Dim, scalar>& t)
    {
        this->len = 1;
        for (int d = 0; d < Dim; ++d)
        {
            this->_shape[d] = t._shape[d];
            this->len *= this->_shape[d];
        }
        mem.reset(new scalar[this->len]);
        this->ptr = mem.get();
        
        for (int i = 0; i < this->len; ++i)
            mem[i] = t[i];
    }

    template <int Dim, typename scalar>
    Tensor<Dim, scalar>& Tensor<Dim, scalar>::operator=(const Tensor<Dim, scalar>& t)
    {
        if (this->len != t.len)
        {
            this->len = t.len;
            mem.reset(new scalar[this->len]);
            this->ptr = mem.get();
        }
        for (int i=0; i < Dim; ++i)
            this->_shape[i] = t._shape[i];
        for (int i=0; i < this->len; ++i)
            mem[i] = t[i];
        
        return *this;
    }

    /// @brief specialization of `TensorWrapper` when `Dim == 1`
    template <typename scalar>
    using VectorWrapper = TensorWrapper<1, scalar>;

    /// @brief specialization of `TensorWrapper` when `Dim == 2`
    template <typename scalar>
    using MatrixWrapper = TensorWrapper<2, scalar>;

    /// @brief specialization of `TensorWrapper` when `Dim == 3`
    template <typename scalar>
    using CubeWrapper = TensorWrapper<3, scalar>;

    /// @brief specialization of `TensorWrapper` when `Dim == 1` and `scalar == double`.
    typedef TensorWrapper<1, double> dvec_wrapper;

    /// @brief specialization of `TensorWrapper` when `Dim == 1` and `scalar == const double`.
    typedef TensorWrapper<1, const double> const_dvec_wrapper;

    /// @brief specialization of `TensorWrapper` when `Dim == 2` and `scalar == double`.
    typedef TensorWrapper<2, double> dmat_wrapper;

    /// @brief specialization of `TensorWrapper` when `Dim == 2` and `scalar == const double`.
    typedef TensorWrapper<2, const double> const_dmat_wrapper;

    /// @brief specialization of `TensorWrapper` when `Dim == 3` and `scalar == double`.
    typedef TensorWrapper<3, double> dcube_wrapper;

    /// @brief specialization of `TensorWrapper` when `Dim == 3` and `scalar == const double`.
    typedef TensorWrapper<3, const double> const_dcube_wrapper;

    /// @brief specialization of `TensorWrapper` when `Dim == 1` and `scalar == int`.
    typedef TensorWrapper<1, int> ivec_wrapper;

    /// @brief specialization of `TensorWrapper` when `Dim == 1` and `scalar == const int`.
    typedef TensorWrapper<1, const int> const_ivec_wrapper;

    /// @brief specialization of `TensorWrapper` when `Dim == 2` and `scalar == int`.
    typedef TensorWrapper<2, int> imat_wrapper;

    /// @brief specialization of `TensorWrapper` when `Dim == 2` and `scalar == const int`.
    typedef TensorWrapper<2, const int> const_imat_wrapper;

    /// @brief specialization of `TensorWrapper` when `Dim == 3` and `scalar == int`.
    typedef TensorWrapper<3, int> icube_wrapper;

    /// @brief specialization of `TensorWrapper` when `Dim == 3` and `scalar == const int`.
    typedef TensorWrapper<3, const int> const_icube_wrapper;

    /// @brief specialization of `Tensor` when `Dim == 1`
    template <typename scalar>
    using Vec = Tensor<1, scalar>;

    /// @brief specialization of `Tensor` when `Dim == 2`
    template <typename scalar>
    using Matrix = Tensor<2, scalar>;

    /// @brief specialization of `Tensor` when `Dim == 3`
    template <typename scalar>
    using Cube = Tensor<3, scalar>;

    /// @brief specialization of `Tensor` when `Dim == 1` and `scalar == double`.
    typedef Vec<double> dvec;

    /// @brief specialization of `Tensor` when `Dim == 2` and `scalar == double`.
    typedef Matrix<double> dmat;

    /// @brief specialization of `Tensor` when `Dim == 3` and `scalar == double`.
    typedef Cube<double> dcube;

    /// @brief specialization of `Tensor` when `Dim == 1` and `scalar == int`.
    typedef Vec<int> ivec;

    /// @brief specialization of `Tensor` when `Dim == 2` and `scalar == int`.
    typedef Matrix<int> imat;

    /// @brief specialization of `Tensor` when `Dim == 3` and `scalar == int`.
    typedef Cube<int> icube;
} // namespace cuddh

#endif