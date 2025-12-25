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
        .def_prop_ro("strides", &ttcore::TensorImpl::strides)
        .def_prop_ro("ndim", &ttcore::TensorImpl::ndim)
        .def_prop_ro("dtype", &ttcore::TensorImpl::dtype)
        .def_prop_ro("device", &ttcore::TensorImpl::device)
        .def_prop_ro("T", &ttcore::TensorImpl::T)

        .def("numel", &ttcore::TensorImpl::numel)
        .def("is_contiguous", &ttcore::TensorImpl::is_contiguous)
        .def("view", &ttcore::TensorImpl::view)
        .def("transpose", &ttcore::TensorImpl::transpose)
        .def("contiguous", &ttcore::TensorImpl::contiguous)
        .def("clone", &ttcore::TensorImpl::clone)

        .def("__getitem__",
             [](const ttcore::TensorImpl& self, const std::vector<int64_t>& indices) {
                 return self.get(indices);
             })
        .def("__setitem__",
             [](ttcore::TensorImpl& self, const std::vector<int64_t>& indices, double value) {
                 self.set(indices, value);
             });
}
