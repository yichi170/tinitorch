#ifndef TT_CORE_TENSOR_IMPL_H
#define TT_CORE_TENSOR_IMPL_H

#include "ttcore/storage.h"

#include <cstdint>
#include <vector>

namespace ttcore {
class TensorImpl {
public:
    TensorImpl(std::vector<int64_t> shape, DType dtype, Device device);
    ~TensorImpl();

    std::vector<int64_t> shape() const { return shape_; }
    DType dtype() const { return dtype_; }
    Device device() const { return device_; }

    // Storage* storage();
    // const Storage* storage() const;

private:
    std::vector<int64_t> shape_;
    DType dtype_;
    Device device_;
    std::shared_ptr<Storage> storage_;
};
}  // end namespace ttcore
#endif  // TT_CORE_TENSOR_IMPL_H
