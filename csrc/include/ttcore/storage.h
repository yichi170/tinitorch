#ifndef TT_CORE_STORAGE_H
#define TT_CORE_STORAGE_H

#include "ttcore/device.h"
#include "ttcore/dtype.h"

#include <cstdlib>
#include <memory>

namespace ttcore {
class Storage {
public:
    Storage(size_t size, DType dtype, Device device) : size_(size), dtype_(dtype), device_(device) {
        data_ = std::malloc(size * dtype_size(dtype));
    };
    ~Storage() { std::free(data_); }

    Storage(const Storage& other) = delete;
    Storage& operator=(const Storage& other) = delete;

    void* data() { return data_; }
    const void* data() const { return data_; }
    size_t size() const { return size_; }
    DType dtype() const { return dtype_; }
    Device device() const { return device_; }

    template <typename T>
    T* data_as() {
        return static_cast<T*>(data_);
    }

    template <typename T>
    const T* data_as() const {
        return static_cast<const T*>(data_);
    }

    template <typename T>
    void set(size_t idx, T value) {
        data_as<T>()[idx] = value;
    }

    template <typename T>
    T get(size_t idx) const {
        return data_as<T>()[idx];
    }

    template <typename T>
    void fill(T value) {
        for (size_t i = 0; i < size_; i++) {
            set(i, value);
        }
    }

private:
    void* data_;
    size_t size_;
    DType dtype_;
    Device device_;
};
}  // end namespace ttcore

#endif  // TT_CORE_STORAGE_H
