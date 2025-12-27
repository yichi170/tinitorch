#include "ttcore/kernels.h"

#include <stdexcept>

namespace ttcore {
namespace kernels {

TensorImpl cpu_relu_f32(const std::vector<TensorImpl*>& inputs) {
    if (inputs.size() != 1) {
        throw std::runtime_error("relu requires exactly 1 input");
    }

    const TensorImpl* x = inputs[0];
    TensorImpl result(x->shape(), x->dtype(), x->device());

    const int64_t n = result.numel();
    for (int64_t i = 0; i < n; ++i) {
        double val = x->get_flat(i);
        result.set_flat(i, val > 0.0 ? val : 0.0);
    }

    return result;
}

void register_cpu_activation_kernels(Dispatcher& registry) {
    registry.add_kernel("relu", Device::CPU, DType::Float32, cpu_relu_f32);
}

}  // end namespace kernels
}  // end namespace ttcore
