#include "ttcore/kernels.h"

#include <stdexcept>

namespace ttcore {
namespace kernels {

namespace {

void check_binary_inputs(const std::vector<TensorImpl*>& inputs, const char* op_name) {
    if (inputs.size() != 2) {
        throw std::runtime_error(std::string(op_name) + " requires exactly 2 inputs");
    }
    if (inputs[0]->shape() != inputs[1]->shape()) {
        throw std::runtime_error(std::string(op_name) +
                                 " requires inputs with the same shape (no broadcasting yet)");
    }
}

void check_unary_input(const std::vector<TensorImpl*>& inputs, const char* op_name) {
    if (inputs.size() != 1) {
        throw std::runtime_error(std::string(op_name) + " requires exactly 1 input");
    }
}

template <typename BinaryOp>
TensorImpl binary_elementwise(const TensorImpl* a, const TensorImpl* b, BinaryOp op) {
    TensorImpl result(a->shape(), a->dtype(), a->device());
    const int64_t n = result.numel();

    // TODO: optimize with direct pointer access when contiguous
    for (int64_t i = 0; i < n; ++i) {
        result.set_flat(i, op(a->get_flat(i), b->get_flat(i)));
    }
    return result;
}

template <typename UnaryOp>
TensorImpl unary_elementwise(const TensorImpl* a, UnaryOp op) {
    TensorImpl result(a->shape(), a->dtype(), a->device());
    const int64_t n = result.numel();

    for (int64_t i = 0; i < n; ++i) {
        result.set_flat(i, op(a->get_flat(i)));
    }
    return result;
}

}  // end namespace

TensorImpl cpu_add_f32(const std::vector<TensorImpl*>& inputs) {
    check_binary_inputs(inputs, "add");
    return binary_elementwise(inputs[0], inputs[1], [](double a, double b) {
        return a + b;
    });
}

TensorImpl cpu_sub_f32(const std::vector<TensorImpl*>& inputs) {
    check_binary_inputs(inputs, "sub");
    return binary_elementwise(inputs[0], inputs[1], [](double a, double b) {
        return a - b;
    });
}

TensorImpl cpu_mul_f32(const std::vector<TensorImpl*>& inputs) {
    check_binary_inputs(inputs, "mul");
    return binary_elementwise(inputs[0], inputs[1], [](double a, double b) {
        return a * b;
    });
}

TensorImpl cpu_div_f32(const std::vector<TensorImpl*>& inputs) {
    check_binary_inputs(inputs, "div");
    return binary_elementwise(inputs[0], inputs[1], [](double a, double b) {
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
