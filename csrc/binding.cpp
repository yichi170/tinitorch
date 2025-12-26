#include "ttcore/dispatch.h"
#include "ttcore/ops.h"
#include "ttcore/tensor_impl.h"

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;

namespace {

void infer_shape(nb::handle obj, std::vector<int64_t>& shape, size_t depth = 0) {
    if (nb::isinstance<nb::list>(obj)) {
        nb::list lst = nb::cast<nb::list>(obj);
        if (depth >= shape.size()) {
            shape.push_back(static_cast<int64_t>(nb::len(lst)));
        }
        if (nb::len(lst) > 0) {
            infer_shape(lst[0], shape, depth + 1);
        }
    }
}

void flatten_data(nb::handle obj, std::vector<double>& data) {
    if (nb::isinstance<nb::list>(obj)) {
        nb::list lst = nb::cast<nb::list>(obj);
        for (size_t i = 0; i < nb::len(lst); ++i) {
            flatten_data(lst[i], data);
        }
    } else {
        data.push_back(nb::cast<double>(obj));
    }
}

ttcore::TensorImpl from_nested_list(nb::object data, ttcore::DType dtype, ttcore::Device device) {
    std::vector<int64_t> shape;
    std::vector<double> flat_data;

    infer_shape(data, shape);
    flatten_data(data, flat_data);

    return ttcore::TensorImpl(shape, flat_data, dtype, device);
}

}  // namespace

NB_MODULE(_C, m) {
    m.doc() = "TiniTorch C++ core";

    nb::enum_<ttcore::DType>(m, "DType")
        .value("Float32", ttcore::DType::Float32)
        .value("Float64", ttcore::DType::Float64);

    nb::enum_<ttcore::Device>(m, "Device")
        .value("CPU", ttcore::Device::CPU)
        .value("CUDA", ttcore::Device::CUDA);

    nb::class_<ttcore::TensorImpl>(m, "TensorImpl")
        .def(
            "__init__",
            [](ttcore::TensorImpl* self, nb::object data, ttcore::DType dtype,
               ttcore::Device device) {
                new (self) ttcore::TensorImpl(from_nested_list(data, dtype, device));
            },
            nb::arg("data"), nb::arg("dtype") = ttcore::DType::Float32,
            nb::arg("device") = ttcore::Device::CPU)

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

    m.def("dispatch", &ttcore::dispatch);
}
