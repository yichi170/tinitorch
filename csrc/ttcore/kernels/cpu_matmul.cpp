#include "ttcore/kernels.h"

#include <stdexcept>

namespace ttcore {
namespace kernels {

TensorImpl cpu_matmul_f32(const std::vector<TensorImpl*>& inputs) {
    if (inputs.size() != 2) {
        throw std::runtime_error("matmul requires exactly 2 inputs");
    }

    const TensorImpl* a = inputs[0];
    const TensorImpl* b = inputs[1];

    if (a->ndim() != 2 || b->ndim() != 2) {
        throw std::runtime_error("matmul requires 2D tensors");
    }

    const auto& a_shape = a->shape();
    const auto& b_shape = b->shape();

    if (a_shape[1] != b_shape[0]) {
        throw std::runtime_error("matmul: inner dimensions must match");
    }

    const int64_t m = a_shape[0];
    const int64_t k = a_shape[1];
    const int64_t n = b_shape[1];

    TensorImpl result({m, n}, a->dtype(), a->device());

    for (int64_t i = 0; i < m; ++i) {
        for (int64_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (int64_t p = 0; p < k; ++p) {
                sum += a->get({i, p}) * b->get({p, j});
            }
            result.set({i, j}, sum);
        }
    }

    return result;
}

void register_cpu_matmul_kernels(Dispatcher& registry) {
    registry.add_kernel("matmul", Device::CPU, DType::Float32, cpu_matmul_f32);
}

}  // end namespace kernels
}  // end namespace ttcore
