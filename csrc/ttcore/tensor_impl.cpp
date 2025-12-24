#include "ttcore/tensor_impl.h"

#include <iostream>

namespace ttcore {

TensorImpl::TensorImpl(std::vector<int64_t> shape, DType dtype, Device device)
    : shape_(shape), dtype_(dtype), device_(device) {
    size_t numel = 1;
    for (auto dim : shape) {
        numel *= static_cast<size_t>(dim);
    }
    storage_ = std::make_shared<Storage>(numel, dtype, device);

    // Simple test, TODO: remove it!
    storage_->fill(3.0f);

    for (size_t i = 0; i < storage_->size(); i++) {
        std::cout << storage_->get<float>(i) << " ";
    }
    std::cout << std::endl;
}

TensorImpl::~TensorImpl() = default;

}  // end namespace ttcore
