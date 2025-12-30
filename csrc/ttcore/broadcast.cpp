#include "ttcore/broadcast.h"

#include <algorithm>
#include <stdexcept>

namespace ttcore {

bool is_broadcastable(const std::vector<int64_t>& a, const std::vector<int64_t>& b) {
    size_t ndim = std::max(a.size(), b.size());

    for (size_t i = 0; i < ndim; ++i) {
        int64_t dim_a = (i < ndim - a.size()) ? 1 : a[i - (ndim - a.size())];
        int64_t dim_b = (i < ndim - b.size()) ? 1 : b[i - (ndim - b.size())];

        if (dim_a != dim_b && dim_a != 1 && dim_b != 1) {
            return false;
        }
    }
    return true;
}

std::vector<int64_t> broadcast_shapes(const std::vector<int64_t>& a,
                                      const std::vector<int64_t>& b) {
    size_t ndim = std::max(a.size(), b.size());
    std::vector<int64_t> result(ndim);

    for (size_t i = 0; i < ndim; ++i) {
        // Align from the right: pad shorter shape with 1s on the left
        int64_t dim_a = (i < ndim - a.size()) ? 1 : a[i - (ndim - a.size())];
        int64_t dim_b = (i < ndim - b.size()) ? 1 : b[i - (ndim - b.size())];

        if (dim_a == dim_b) {
            result[i] = dim_a;
        } else if (dim_a == 1) {
            result[i] = dim_b;
        } else if (dim_b == 1) {
            result[i] = dim_a;
        } else {
            throw std::runtime_error("Cannot broadcast: incompatible dimensions");
        }
    }
    return result;
}

std::vector<int64_t> broadcast_strides(const std::vector<int64_t>& shape,
                                       const std::vector<int64_t>& strides,
                                       const std::vector<int64_t>& target_shape) {
    size_t ndim = target_shape.size();
    size_t offset = ndim - shape.size();
    std::vector<int64_t> result(ndim, 0);  // Initialize with 0 (broadcast dim)

    for (size_t i = 0; i < shape.size(); ++i) {
        size_t target_idx = offset + i;
        if (shape[i] == target_shape[target_idx]) {
            // Same size: use original stride
            result[target_idx] = strides[i];
        } else if (shape[i] == 1) {
            // Broadcast: stride = 0 (repeat this element)
            result[target_idx] = 0;
        } else {
            throw std::runtime_error("Cannot broadcast shape to target");
        }
    }
    return result;
}

}  // end namespace ttcore
