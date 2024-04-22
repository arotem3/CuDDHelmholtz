#ifndef CUDDH_HOST_DEVICE_ARRAY_HPP
#define CUDDH_HOST_DEVICE_ARRAY_HPP

#include <utility>
#include <iostream>
#include <cuda_runtime.h>

#include "cuddh_config.hpp"

// manages pointers of arrays that may be accessed between host and device
namespace cuddh
{
    template <typename T>
    class HostDeviceArray
    {
    public:
        // initializes HostDeviceArray. The actual memory is not initialized until a
        // call is made to any of the read/write functions.
        HostDeviceArray(int n);

        // no copy allowed. HostDeviceArray is like unique_ptr.
        HostDeviceArray(const HostDeviceArray&) = delete;
        HostDeviceArray& operator=(const HostDeviceArray&) = delete;

        // move HostDeviceArray
        HostDeviceArray(HostDeviceArray&&);
        HostDeviceArray& operator=(HostDeviceArray&&);

        ~HostDeviceArray();

        // returns the size of the array
        int size() const;

        // returns a read only host pointer to the array. This potentially copies
        // the data from device. (The copy occurs only if the last write access to the
        // memory was by the device. If the memory was previously modified by the
        // host, then no copy occurs)
        const T * host_read(bool force_copy=false) const;

        // returns a host pointer to the array without corroborating the data with
        // the device. This invalidates the device data, so the next call
        // device_read() or device_read_write() will cause a copy from host to
        // device.
        T * host_write();

        // returns a host pointer to the array. This potentially copies the data
        // from the device and also invalidates the device data, so the next call to
        // device_read() or device_read_write() will cause a copy from host to
        // device.
        T * host_read_write(bool force_copy=false);

        // returns the pointer to the host array and releases ownership of it. No
        // checks are made for validity. The memory is returned as is and may be
        // null if never initialized.
        T * host_release();

        // returns a read only device pointer to the array. This potentially copies
        // the data from host. (The copy occurs only if the last write access to the
        // memory was by the host. If the memory was previously modified by the
        // device, then no copy occurs)
        const T * device_read(bool force_copy=false) const;

        // returns a device pointer to the array without corroborating the data with
        // the host. This invalidates the host data, so the next call
        // host_read() or host_read_write() will cause a copy from device to
        // host.
        T * device_write();

        // returns a device pointer to the array. This potentially copies the data
        // from the host and also invalidates the host data, so the next call to
        // host_read() or host_read_write() will cause a copy from device to host.
        T * device_read_write(bool force_copy=false);

        // returns the pointer to the device array and releases ownership of it. No
        // checks are made for validity. The memory is returned as is and may be
        // null if never initialized.
        T * device_release();

    private:
        int n;
        
        mutable bool device_is_valid;
        mutable bool host_is_valid;
        
        mutable T * device_array;
        mutable T * host_array;
    };

    template <typename T>
    HostDeviceArray<T>::HostDeviceArray(int n_) : n{n_}, device_is_valid{false}, host_is_valid{false}, device_array{nullptr}, host_array{nullptr} {}

    template <typename T>
    HostDeviceArray<T>::HostDeviceArray(HostDeviceArray&& x)
        : n{x.n},
        device_is_valid{std::exchange(x.device_is_valid, false)},
        host_is_valid{std::exchange(x.host_is_valid, false)},
        device_array{std::exchange(x.device_array, nullptr)},
        host_array{std::exchange(x.host_array, nullptr)}
    {}

    template <typename T>
    HostDeviceArray<T>& HostDeviceArray<T>::operator=(HostDeviceArray&& x)
    {
        n = x.n;
        device_is_valid = std::exchange(x.device_is_valid, false);
        host_is_valid = std::exchange(x.host_is_valid, false);
        device_array = std::exchange(x.device_array, nullptr);
        host_array = std::exchange(x.host_array, nullptr);
        
        return *this;
    }

    template <typename T>
    HostDeviceArray<T>::~HostDeviceArray()
    {
        delete[] host_array;
        cudaFree(device_array);
    }

    template <typename T>
    int HostDeviceArray<T>::size() const
    {
        return n;
    }

    template <typename T>
    const T * HostDeviceArray<T>::host_read(bool force_copy) const
    {
    #ifdef CUDDH_LOG_MEMCPY
        std::cout << "host read:" << std::endl;
    #endif
        if (not host_is_valid || force_copy) // check if data on host is current (or force copy)
        {
            const int size = n * sizeof(T);
            if (not host_array) // memory never initialized, need to allocate
            {
            #ifdef CUDDH_LOG_MEMCPY
                std::cout << "\tallocating new host array (" << size << " bytes)." << std::endl;
            #endif
                host_array = new T[n];
            }
            
            if (device_is_valid) // device data is most current, need to copy
            {   
            #ifdef CUDDH_LOG_MEMCPY
                std::cout << "\tcopying data D <- H (" << size << " bytes)." << std::endl;
            #endif
                cudaMemcpy(host_array, device_array, size, cudaMemcpyDeviceToHost);
            }
        }

    #ifdef CUDDH_LOG_MEMCPY
        std::cout << "\treturning host array: " << host_array << std::endl;
    #endif
        
        host_is_valid = true; // host is most current
        return host_array;
    }

    template <typename T>
    T * HostDeviceArray<T>::host_write()
    {
    #ifdef CUDDH_LOG_MEMCPY
        std::cout << "host write:" << std::endl;
    #endif
        if (not host_array) // memory never allocated. Need to initialize
        {
        #ifdef CUDDH_LOG_MEMCPY
            std::cout << "\tallocating new host array (" << (n*sizeof(T)) << " bytes)." << std::endl;
        #endif
            host_array = new T[n];
        }

    #ifdef CUDDH_LOG_MEMCPY
        std::cout << "\treturning host array: " << host_array << std::endl;
    #endif
        
        host_is_valid = true; // host is most current
        device_is_valid = false; // device is outdated

        return host_array;
    }

    template <typename T>
    T * HostDeviceArray<T>::host_read_write(bool force_copy)
    {
        host_read(force_copy);
        return host_write();
    }

    template <typename T>
    T * HostDeviceArray<T>::host_release()
    {
        T * h_a = host_array;
        host_array = nullptr;
        return h_a;
    }

    template <typename T>
    const T * HostDeviceArray<T>::device_read(bool force_copy) const
    {
    #ifdef CUDDH_LOG_MEMCPY
        std::cout << "Device read:" << std::endl;
    #endif
        if (not device_is_valid || force_copy) // check if device data is current (or force copy)
        {
            const int size = n * sizeof(T);
            if (not device_array) // device memory never initialized.
            {
            #ifdef CUDDH_LOG_MEMCPY
                std::cout << "\tallocating new device array (" << size << " bytes)" << std::endl;
            #endif
                cudaMalloc(&device_array, size);
            }
            if (host_is_valid) // host data is most current, need to copy
            {
            #ifdef CUDDH_LOG_MEMCPY
                std::cout << "\tcopying data D <- H (" << size << " bytes)" << std::endl;
            #endif
                cudaMemcpy(device_array, host_array, size, cudaMemcpyHostToDevice);
            }
        }

    #ifdef CUDDH_LOG_MEMCPY
        std::cout << "\treturning device array: " << device_array << std::endl;
    #endif

        device_is_valid = true; // device data is now current
        return device_array;
    }

    template <typename T>
    T * HostDeviceArray<T>::device_write()
    {
    #ifdef CUDDH_LOG_MEMCPY
        std::cout << "Device write:" << std::endl;
    #endif
        if (not device_array) // device memory never initialized
        {
            const int size = n * sizeof(T);
        #ifdef CUDDH_LOG_MEMCPY
            std::cout << "\tallocating new device array (" << size << " bytes)" << std::endl;
        #endif
            cudaMalloc(&device_array, size);
        }

    #ifdef CUDDH_LOG_MEMCPY
        std::cout << "\treturning device array: " << device_array << std::endl;
    #endif

        device_is_valid = true; // device is most current
        host_is_valid = false; // host is now outdated

        return device_array;
    }

    template <typename T>
    T * HostDeviceArray<T>::device_read_write(bool force_copy)
    {
        device_read(force_copy);
        return device_write();
    }

    template <typename T>
    T * HostDeviceArray<T>::device_release()
    {
        T * d_a = device_array;
        device_array = nullptr;
        return d_a;
    }

    typedef HostDeviceArray<double> host_device_dvec;
    typedef HostDeviceArray<int> host_device_ivec;

} // namespace cuddh

#endif
