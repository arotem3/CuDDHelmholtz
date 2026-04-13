#pragma once

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <utility>

#include "cuddh_config.hpp"

// manages pointers of arrays that may be accessed between host and device
namespace cuddh
{
    enum class MemorySpace
    {
        HOST,
        DEVICE
    };

    template <typename T>
    class HostDeviceArray
    {
    public:
        // initializes HostDeviceArray. The actual memory is not initialized until a
        // call is made to any of the read/write functions.
        HostDeviceArray(int n);

        // initialize empty array
        HostDeviceArray();

        HostDeviceArray(const HostDeviceArray &) = default;
        HostDeviceArray &operator=(const HostDeviceArray &) = default;

        // move HostDeviceArray
        HostDeviceArray(HostDeviceArray &&);
        HostDeviceArray &operator=(HostDeviceArray &&);

        ~HostDeviceArray() = default;

        // returns the size of the array
        constexpr int size() const { return n; }

        // resizes the array and invalidates both host and device pointer. This
        // action deletes previous data.
        void resize(int new_size);

        // read access to data, same as host_read or device_read for respective MemorySpace.
        const T *read(MemorySpace m) const;

        // write access to data, same as host_write or device_write for respective MemorySpace.
        T *write(MemorySpace m);

        // read & write access to data, same as host_read_write or
        // device_read_write for respective MemorySpace.
        T *read_write(MemorySpace m);

        // returns a read only host pointer to the array. This potentially copies
        // the data from device. (The copy occurs only if the last write access to the
        // memory was by the device. If the memory was previously modified by the
        // host, then no copy occurs)
        const T *host_read() const;

        // returns a host pointer to the array without corroborating the data with
        // the device. This invalidates the device data, so the next call
        // device_read() or device_read_write() will cause a copy from host to
        // device.
        T *host_write();

        // returns a host pointer to the array. This potentially copies the data
        // from the device and also invalidates the device data, so the next call to
        // device_read() or device_read_write() will cause a copy from host to
        // device.
        T *host_read_write();

        // returns a read only device pointer to the array. This potentially copies
        // the data from host. (The copy occurs only if the last write access to the
        // memory was by the host. If the memory was previously modified by the
        // device, then no copy occurs)
        const T *device_read() const;

        // returns a device pointer to the array without corroborating the data with
        // the host. This invalidates the host data, so the next call
        // host_read() or host_read_write() will cause a copy from device to
        // host.
        T *device_write();

        // returns a device pointer to the array. This potentially copies the data
        // from the host and also invalidates the host data, so the next call to
        // host_read() or host_read_write() will cause a copy from device to host.
        T *device_read_write();

    private:
        int n;

        mutable bool device_is_valid;
        mutable bool host_is_valid;

        mutable thrust::device_vector<T> device_array;
        mutable thrust::host_vector<T> host_array;
    };

    template <typename T>
    HostDeviceArray<T>::HostDeviceArray(int n_)
        : n{n_}, device_is_valid{false}, host_is_valid{false}, device_array{}, host_array{}
    {}

    template <typename T>
    HostDeviceArray<T>::HostDeviceArray()
        : n{0}, device_is_valid{false}, host_is_valid{false}, device_array{}, host_array{}
    {}

    template <typename T>
    HostDeviceArray<T>::HostDeviceArray(HostDeviceArray &&x)
        : n{x.n},
          device_is_valid{std::exchange(x.device_is_valid, false)},
          host_is_valid{std::exchange(x.host_is_valid, false)},
          device_array{std::move(x.device_array)},
          host_array{std::move(x.host_array)}
    {}

    template <typename T>
    HostDeviceArray<T> &HostDeviceArray<T>::operator=(HostDeviceArray &&x)
    {
        n = x.n;
        device_is_valid = std::exchange(x.device_is_valid, false);
        host_is_valid = std::exchange(x.host_is_valid, false);
        device_array = std::move(x.device_array);
        host_array = std::move(x.host_array);

        return *this;
    }

    template <typename T>
    void HostDeviceArray<T>::resize(int new_size)
    {
        host_array.clear();
        device_array.clear();

        host_is_valid = false;
        device_is_valid = false;

        n = new_size;
    }

    template <typename T>
    const T *HostDeviceArray<T>::host_read() const
    {
        if (n < 1)
            return nullptr;

        if (host_array.size() != static_cast<size_t>(n))
            host_array.assign(n, T{});

        if (not host_is_valid && device_is_valid)
            host_array = device_array;

        host_is_valid = true;
        return host_array.data();
    }

    template <typename T>
    T *HostDeviceArray<T>::host_write()
    {
        if (n < 1)
            return nullptr;

        if (host_array.size() != static_cast<size_t>(n))
            host_array.assign(n, T{});

        host_is_valid = true;
        device_is_valid = false;

        return host_array.data();
    }

    template <typename T>
    T *HostDeviceArray<T>::host_read_write()
    {
        host_read();
        return host_write();
    }

    template <typename T>
    const T *HostDeviceArray<T>::device_read() const
    {
        if (n < 1)
            return nullptr;

        if (device_array.size() != static_cast<size_t>(n))
            device_array.assign(n, T{});

        if (not device_is_valid && host_is_valid)
            device_array = host_array;

        device_is_valid = true;
        return thrust::raw_pointer_cast(device_array.data());
    }

    template <typename T>
    T *HostDeviceArray<T>::device_write()
    {
        if (n < 1)
            return nullptr;

        if (device_array.size() != static_cast<size_t>(n))
            device_array.assign(n, T{});

        device_is_valid = true;
        host_is_valid = false;

        return thrust::raw_pointer_cast(device_array.data());
    }

    template <typename T>
    T *HostDeviceArray<T>::device_read_write()
    {
        device_read();
        return device_write();
    }

    template <typename T>
    const T *HostDeviceArray<T>::read(MemorySpace m) const
    {
        if (m == MemorySpace::HOST)
            return host_read();
        else
            return device_read();
    }

    template <typename T>
    T *HostDeviceArray<T>::write(MemorySpace m)
    {
        if (m == MemorySpace::HOST)
            return host_write();
        else
            return device_write();
    }

    template <typename T>
    T *HostDeviceArray<T>::read_write(MemorySpace m)
    {
        if (m == MemorySpace::HOST)
            return host_read_write();
        else
            return device_read_write();
    }

    typedef HostDeviceArray<double> host_device_dvec;
    typedef HostDeviceArray<int> host_device_ivec;

} // namespace cuddh
