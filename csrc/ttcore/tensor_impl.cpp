#include "ttcore/tensor_impl.h"

#include <stdexcept>

namespace ttcore {

std::vector<int64_t> TensorImpl::compute_strides(const std::vector<int64_t>& shape) {
    if (shape.empty()) {
        return {};
    }
    std::vector<int64_t> strides(shape.size(), 1);
    // Use int64_t to avoid underflow when size is 1. size_t is unsigned.
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    return strides;
}

static int64_t compute_numel(const std::vector<int64_t>& shape) {
    int64_t n = 1;
    for (int64_t dim : shape) {
        n *= dim;
    }
    return n;
}

TensorImpl::TensorImpl(std::vector<int64_t> shape, DType dtype, Device device)
    : shape_(std::move(shape)),
      strides_(compute_strides(shape_)),
      offset_(0),
      dtype_(dtype),
      device_(device) {
    numel_ = compute_numel(shape_);
    storage_ = std::make_shared<Storage>(static_cast<size_t>(numel_), dtype, device);
}

TensorImpl::TensorImpl(std::vector<int64_t> shape, const std::vector<double>& data, DType dtype,
                       Device device)
    : shape_(std::move(shape)),
      strides_(compute_strides(shape_)),
      offset_(0),
      dtype_(dtype),
      device_(device) {
    numel_ = compute_numel(shape_);
    if (static_cast<int64_t>(data.size()) != numel_) {
        throw std::runtime_error("data size does not match shape");
    }
    storage_ = std::make_shared<Storage>(static_cast<size_t>(numel_), dtype, device);
    for (int64_t i = 0; i < numel_; ++i) {
        set_flat(i, data[i]);
    }
}

TensorImpl::TensorImpl(std::shared_ptr<Storage> storage, std::vector<int64_t> shape,
                       std::vector<int64_t> strides, int64_t offset, DType dtype, Device device)
    : storage_(std::move(storage)),
      shape_(std::move(shape)),
      strides_(std::move(strides)),
      offset_(offset),
      dtype_(dtype),
      device_(device) {
    numel_ = compute_numel(shape_);
}

TensorImpl::~TensorImpl() = default;

bool TensorImpl::is_contiguous() const {
    if (shape_.empty()) {
        return true;
    }
    std::vector<int64_t> expected = compute_strides(shape_);
    return strides_ == expected;
}

int64_t TensorImpl::compute_flat_index(const std::vector<int64_t>& indices) const {
    int64_t idx = offset_;
    for (size_t i = 0; i < indices.size(); ++i) {
        idx += indices[i] * strides_[i];
    }
    return idx;
}

double TensorImpl::get_flat(int64_t idx) const {
    switch (dtype_) {
        case DType::Float32:
            return static_cast<double>(storage_->get<float>(idx));
        case DType::Float64:
            return storage_->get<double>(idx);
        default:
            throw std::runtime_error("Unknown dtype");
    }
}

void TensorImpl::set_flat(int64_t idx, double value) {
    switch (dtype_) {
        case DType::Float32:
            storage_->set<float>(idx, static_cast<float>(value));
            break;
        case DType::Float64:
            storage_->set<double>(idx, value);
            break;
        default:
            throw std::runtime_error("Unknown dtype");
    }
}

double TensorImpl::get(const std::vector<int64_t>& indices) const {
    int64_t idx = compute_flat_index(indices);
    return get_flat(idx);
}

void TensorImpl::set(const std::vector<int64_t>& indices, double value) {
    int64_t idx = compute_flat_index(indices);
    set_flat(idx, value);
}

TensorImpl TensorImpl::view(std::vector<int64_t> new_shape) const {
    if (!is_contiguous()) {
        throw std::runtime_error("view requires contiguous tensor");
    }

    // Handle -1 dimension (infer one dimension)
    int64_t infer_idx = -1;
    int64_t known_numel = 1;
    for (size_t i = 0; i < new_shape.size(); ++i) {
        if (new_shape[i] == -1) {
            if (infer_idx != -1) {
                throw std::runtime_error("only one -1 allowed in view");
            }
            infer_idx = static_cast<int64_t>(i);
        } else {
            known_numel *= new_shape[i];
        }
    }

    if (infer_idx != -1) {
        if (numel_ % known_numel != 0) {
            throw std::runtime_error("shape not compatible with numel");
        }
        new_shape[infer_idx] = numel_ / known_numel;
    }

    int64_t new_numel = 1;
    for (int64_t dim : new_shape) {
        new_numel *= dim;
    }
    if (new_numel != numel_) {
        throw std::runtime_error("numel mismatch in view");
    }

    return TensorImpl(storage_, new_shape, compute_strides(new_shape), offset_, dtype_, device_);
}

TensorImpl TensorImpl::transpose(int64_t dim0, int64_t dim1) const {
    if (dim0 < 0 || dim0 >= ndim() || dim1 < 0 || dim1 >= ndim()) {
        throw std::runtime_error("transpose dimension out of range");
    }

    std::vector<int64_t> new_shape = shape_;
    std::swap(new_shape[dim0], new_shape[dim1]);
    std::vector<int64_t> new_strides = strides_;
    std::swap(new_strides[dim0], new_strides[dim1]);
    return TensorImpl(storage_, new_shape, new_strides, offset_, dtype_, device_);
}

TensorImpl TensorImpl::T() const {
    if (ndim() < 2) {
        throw std::runtime_error("T requires at least 2 dimensions");
    }
    return transpose(ndim() - 2, ndim() - 1);
}

TensorImpl TensorImpl::contiguous() const {
    if (is_contiguous()) {
        return *this;
    }

    // Allocate new storage and copy elements in row-major order
    TensorImpl result(shape_, dtype_, device_);
    std::vector<int64_t> indices(shape_.size(), 0);

    for (int64_t i = 0; i < numel_; ++i) {
        result.set_flat(i, get(indices));

        for (int64_t d = static_cast<int64_t>(shape_.size()) - 1; d >= 0; --d) {
            indices[d]++;
            if (indices[d] < shape_[d]) {
                break;
            }
            indices[d] = 0;
        }
    }

    return result;
}

TensorImpl TensorImpl::clone() const {
    TensorImpl result(shape_, dtype_, device_);

    if (is_contiguous()) {
        for (int64_t i = 0; i < numel_; ++i) {
            result.set_flat(i, get_flat(offset_ + i));
        }
    } else {
        std::vector<int64_t> indices(shape_.size(), 0);

        for (int64_t i = 0; i < numel_; ++i) {
            result.set_flat(i, get(indices));

            for (int64_t d = static_cast<int64_t>(shape_.size()) - 1; d >= 0; --d) {
                indices[d]++;
                if (indices[d] < shape_[d]) {
                    break;
                }
                indices[d] = 0;
            }
        }
    }

    return result;
}

}  // namespace ttcore
