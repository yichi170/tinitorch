#include "ttcore/ops.h"
#include "ttcore/tensor_impl.h"

#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;

NB_MODULE(_C, m) {
    m.doc() = "TiniTorch C++ core";

    nb::enum_<ttcore::DType>(m, "DType")
        .value("Int32", ttcore::DType::Int32)
        .value("Int64", ttcore::DType::Int64)
        .value("Float32", ttcore::DType::Float32)
        .value("Float64", ttcore::DType::Float64);

    nb::enum_<ttcore::Device>(m, "Device")
        .value("CPU", ttcore::Device::CPU)
        .value("CUDA", ttcore::Device::CUDA);

    nb::class_<ttcore::TensorImpl>(m, "TensorImpl")
        .def(nb::init<std::vector<int64_t>, ttcore::DType, ttcore::Device>())
        .def_prop_ro("shape", &ttcore::TensorImpl::shape)
        .def_prop_ro("dtype", &ttcore::TensorImpl::dtype)
        .def_prop_ro("device", &ttcore::TensorImpl::device);
}
