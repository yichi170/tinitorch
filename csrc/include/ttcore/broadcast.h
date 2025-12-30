#ifndef TT_CORE_BROADCAST_H
#define TT_CORE_BROADCAST_H

#include <cstdint>
#include <vector>

namespace ttcore {

// Compute the broadcasted output shape from two input shapes.
// Throws if shapes are not broadcastable.
std::vector<int64_t> broadcast_shapes(const std::vector<int64_t>& a, const std::vector<int64_t>& b);

// Compute strides for broadcasting a tensor to target_shape.
// For dimensions that are broadcast (size 1 -> size N), stride becomes 0.
std::vector<int64_t> broadcast_strides(const std::vector<int64_t>& shape,
                                       const std::vector<int64_t>& strides,
                                       const std::vector<int64_t>& target_shape);

bool is_broadcastable(const std::vector<int64_t>& a, const std::vector<int64_t>& b);

}  // end namespace ttcore

#endif  // TT_CORE_BROADCAST_H
