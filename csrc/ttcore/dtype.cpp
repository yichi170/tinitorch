#include "ttcore/dtype.h"

namespace ttcore {

size_t dtype_size(DType dtype) {
    switch (dtype) {
        case DType::Float32: return 4;
        case DType::Float64: return 8;
        case DType::Int32: return 4;
        case DType::Int64: return 8;
    }
    return 0;
}

const char* dtype_name(DType dtype) {
    switch (dtype) {
        case DType::Float32: return "float32";
        case DType::Float64: return "float64";
        case DType::Int32: return "int32";
        case DType::Int64: return "int64";
    }
    return "unknown";
}

}  // end namespace ttcore
