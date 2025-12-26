#ifndef TT_CORE_DTYPE_H
#define TT_CORE_DTYPE_H

#include <cstddef>

namespace ttcore {

enum class DType {
    Float32,
    Float64,
};

size_t dtype_size(DType dtype);
const char* dtype_name(DType dtype);

}  // end namespace ttcore

#endif  // TT_CORE_DTYPE_H
