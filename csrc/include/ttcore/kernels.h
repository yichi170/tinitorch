#ifndef TT_CORE_KERNELS_H
#define TT_CORE_KERNELS_H

#include "ttcore/dispatch.h"
#include "ttcore/tensor_impl.h"

namespace ttcore {
namespace kernels {

TensorImpl cpu_add_f32(const std::vector<TensorImpl*>& inputs);
TensorImpl cpu_sub_f32(const std::vector<TensorImpl*>& inputs);
TensorImpl cpu_mul_f32(const std::vector<TensorImpl*>& inputs);
TensorImpl cpu_div_f32(const std::vector<TensorImpl*>& inputs);

TensorImpl cpu_neg_f32(const std::vector<TensorImpl*>& inputs);

void register_cpu_elementwise_kernels(Dispatcher& registry);

}  // end namespace kernels
}  // end namespace ttcore

#endif  // TT_CORE_KERNELS_H
