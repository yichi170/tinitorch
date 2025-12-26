#include "ttcore/dispatch.h"

namespace ttcore {
Dispatcher registry;

TensorImpl dispatch(const std::string& op, Device device, DType dtype,
                    const std::vector<TensorImpl*>& inputs) {
    return registry.dispatch(DispatchKey{op, device, dtype}, inputs);
}

void Dispatcher::add_kernel(std::string op, Device device, DType dtype, KernelFunc func) {
    kernels.emplace(DispatchKey{std::move(op), device, dtype}, std::move(func));
}

TensorImpl Dispatcher::dispatch(const DispatchKey& key,
                                const std::vector<TensorImpl*>& inputs) const {
    auto it = kernels.find(key);
    if (it == kernels.end()) {
        throw std::runtime_error("No kernel registered for key: " + key.op + " " +
                                 dtype_name(key.dtype));
    }
    return it->second(inputs);
}

static void register_kernels() {
    registry.add_kernel("add", Device::CPU, DType::Float32,
                        [](const std::vector<TensorImpl*>& inputs) {
                            if (inputs.size() != 2) {
                                throw std::runtime_error("add requires exactly 2 inputs");
                            }
                            TensorImpl result = inputs[0]->clone();
                            if (inputs[0]->shape() != inputs[1]->shape()) {
                                throw std::runtime_error("add requires inputs with the same shape");
                            }
                            for (int64_t i = 0; i < result.numel(); i++) {
                                result.set_flat(i, inputs[0]->get_flat(i) + inputs[1]->get_flat(i));
                            }
                            return result;
                        });
}

struct KernelRegistrar {
    KernelRegistrar() { register_kernels(); }
};

static KernelRegistrar registrar;
}  // end namespace ttcore
