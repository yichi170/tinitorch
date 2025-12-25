#ifndef TT_CORE_TENSOR_IMPL_H
#define TT_CORE_TENSOR_IMPL_H

#include "ttcore/storage.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace ttcore {

class TensorImpl {
public:
    TensorImpl(std::vector<int64_t> shape, DType dtype, Device device);

    TensorImpl(std::vector<int64_t> shape, const std::vector<double>& data, DType dtype,
               Device device);

    // Create view (shares storage)
    TensorImpl(std::shared_ptr<Storage> storage, std::vector<int64_t> shape,
               std::vector<int64_t> strides, int64_t offset, DType dtype, Device device);

    ~TensorImpl();

    std::vector<int64_t> shape() const { return shape_; }
    std::vector<int64_t> strides() const { return strides_; }
    int64_t offset() const { return offset_; }
    int64_t ndim() const { return static_cast<int64_t>(shape_.size()); }
    DType dtype() const { return dtype_; }
    Device device() const { return device_; }

    int64_t numel() const { return numel_; }
    bool is_contiguous() const;

    double get_flat(int64_t idx) const;
    void set_flat(int64_t idx, double value);

    double get(const std::vector<int64_t>& indices) const;
    void set(const std::vector<int64_t>& indices, double value);

    TensorImpl view(std::vector<int64_t> new_shape) const;
    TensorImpl transpose(int64_t dim0, int64_t dim1) const;
    TensorImpl T() const;
    TensorImpl contiguous() const;
    TensorImpl clone() const;

private:
    std::shared_ptr<Storage> storage_;
    std::vector<int64_t> shape_;
    std::vector<int64_t> strides_;
    int64_t offset_;
    int64_t numel_;
    DType dtype_;
    Device device_;

    int64_t compute_flat_index(const std::vector<int64_t>& indices) const;
    static std::vector<int64_t> compute_strides(const std::vector<int64_t>& shape);
};

}  // namespace ttcore

#endif  // TT_CORE_TENSOR_IMPL_H
