#include "ttcore/dispatch.h"

#include "ttcore/kernels.h"

#include <stdexcept>

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

static void register_all_kernels() {
    kernels::register_cpu_elementwise_kernels(registry);
    kernels::register_cpu_matmul_kernels(registry);
    kernels::register_cpu_activation_kernels(registry);
}

struct KernelRegistrar {
    KernelRegistrar() { register_all_kernels(); }
};

static KernelRegistrar registrar;

}  // end namespace ttcore
