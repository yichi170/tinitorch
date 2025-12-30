#include "ttcore/broadcast.h"
#include "ttcore/kernels.h"

#include <stdexcept>

namespace ttcore {
namespace kernels {

namespace {

void check_binary_inputs(const std::vector<TensorImpl*>& inputs, const char* op_name) {
    if (inputs.size() != 2) {
        throw std::runtime_error(std::string(op_name) + " requires exactly 2 inputs");
    }
    if (!is_broadcastable(inputs[0]->shape(), inputs[1]->shape())) {
        throw std::runtime_error(std::string(op_name) + ": shapes are not broadcastable");
    }
}

void check_unary_input(const std::vector<TensorImpl*>& inputs, const char* op_name) {
    if (inputs.size() != 1) {
        throw std::runtime_error(std::string(op_name) + " requires exactly 1 input");
    }
}

std::vector<int64_t> flat_to_indices(int64_t flat_idx, const std::vector<int64_t>& shape) {
    std::vector<int64_t> indices(shape.size());
    for (int64_t i = static_cast<int64_t>(shape.size()) - 1; i >= 0; --i) {
        indices[i] = flat_idx % shape[i];
        flat_idx /= shape[i];
    }
    return indices;
}

// Compute flat index using strides (supports broadcast strides with 0)
int64_t compute_broadcast_index(const std::vector<int64_t>& indices,
                                const std::vector<int64_t>& strides, int64_t offset) {
    int64_t idx = offset;
    for (size_t i = 0; i < indices.size(); ++i) {
        idx += indices[i] * strides[i];  // stride=0 means broadcast (repeat)
    }
    return idx;
}

template <typename BinaryOp>
TensorImpl binary_elementwise_broadcast(const TensorImpl* a, const TensorImpl* b, BinaryOp op) {
    auto out_shape = broadcast_shapes(a->shape(), b->shape());

    auto a_strides = broadcast_strides(a->shape(), a->strides(), out_shape);
    auto b_strides = broadcast_strides(b->shape(), b->strides(), out_shape);

    TensorImpl result(out_shape, a->dtype(), a->device());
    const int64_t n = result.numel();

    for (int64_t i = 0; i < n; ++i) {
        auto indices = flat_to_indices(i, out_shape);

        int64_t a_idx = compute_broadcast_index(indices, a_strides, a->offset());
        int64_t b_idx = compute_broadcast_index(indices, b_strides, b->offset());

        double val_a = a->get_flat(a_idx);
        double val_b = b->get_flat(b_idx);
        result.set_flat(i, op(val_a, val_b));
    }
    return result;
}

template <typename UnaryOp>
TensorImpl unary_elementwise(const TensorImpl* a, UnaryOp op) {
    const auto& shape = a->shape();
    TensorImpl result(shape, a->dtype(), a->device());
    const int64_t n = result.numel();

    for (int64_t i = 0; i < n; ++i) {
        auto indices = flat_to_indices(i, shape);
        result.set_flat(i, op(a->get(indices)));
    }
    return result;
}

}  // end namespace

TensorImpl cpu_add_f32(const std::vector<TensorImpl*>& inputs) {
    check_binary_inputs(inputs, "add");
    return binary_elementwise_broadcast(inputs[0], inputs[1], [](double a, double b) {
        return a + b;
    });
}

TensorImpl cpu_sub_f32(const std::vector<TensorImpl*>& inputs) {
    check_binary_inputs(inputs, "sub");
    return binary_elementwise_broadcast(inputs[0], inputs[1], [](double a, double b) {
        return a - b;
    });
}

TensorImpl cpu_mul_f32(const std::vector<TensorImpl*>& inputs) {
    check_binary_inputs(inputs, "mul");
    return binary_elementwise_broadcast(inputs[0], inputs[1], [](double a, double b) {
        return a * b;
    });
}

TensorImpl cpu_div_f32(const std::vector<TensorImpl*>& inputs) {
    check_binary_inputs(inputs, "div");
    return binary_elementwise_broadcast(inputs[0], inputs[1], [](double a, double b) {
        return a / b;
    });
}

TensorImpl cpu_neg_f32(const std::vector<TensorImpl*>& inputs) {
    check_unary_input(inputs, "neg");
    return unary_elementwise(inputs[0], [](double a) {
        return -a;
    });
}

void register_cpu_elementwise_kernels(Dispatcher& registry) {
    registry.add_kernel("add", Device::CPU, DType::Float32, cpu_add_f32);
    registry.add_kernel("sub", Device::CPU, DType::Float32, cpu_sub_f32);
    registry.add_kernel("mul", Device::CPU, DType::Float32, cpu_mul_f32);
    registry.add_kernel("div", Device::CPU, DType::Float32, cpu_div_f32);
    registry.add_kernel("neg", Device::CPU, DType::Float32, cpu_neg_f32);
}

}  // end namespace kernels
}  // end namespace ttcore
