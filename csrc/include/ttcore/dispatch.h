#ifndef TT_CORE_DISPATCH_H
#define TT_CORE_DISPATCH_H

#include "ttcore/device.h"
#include "ttcore/dtype.h"
#include "ttcore/tensor_impl.h"

#include <functional>
#include <string>
#include <unordered_map>

namespace ttcore {
struct DispatchKey {
    std::string op;
    Device device;
    DType dtype;

    bool operator==(const DispatchKey& other) const {
        return op == other.op && device == other.device && dtype == other.dtype;
    }
};

struct DispatchKeyHash {
    size_t operator()(const DispatchKey& k) const noexcept {
        size_t h1 = std::hash<std::string>{}(k.op);
        size_t h2 = std::hash<Device>{}(k.device);
        size_t h3 = std::hash<DType>{}(k.dtype);
        return h1 ^ (h2 << 1) ^ (h3 << 2);
    }
};

using KernelFunc = std::function<TensorImpl(const std::vector<TensorImpl*>&)>;

class Dispatcher {
public:
    void add_kernel(std::string op, Device device, DType dtype, KernelFunc func);
    TensorImpl dispatch(const DispatchKey& key, const std::vector<TensorImpl*>& inputs) const;

private:
    std::unordered_map<DispatchKey, KernelFunc, DispatchKeyHash> kernels;
};

// exposed for binding
TensorImpl dispatch(const std::string& op, Device device, DType dtype,
                    const std::vector<TensorImpl*>& inputs);

}  // end namespace ttcore

#endif  // TT_CORE_DISPATCH_H
