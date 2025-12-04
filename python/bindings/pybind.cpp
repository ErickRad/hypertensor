#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "tensor.h"

namespace py = pybind11;
using namespace hypertensor;

PYBIND11_MODULE(hypertensor, m) {
    py::enum_<DeviceType>(m, "DeviceType")
        .value("CPU", DeviceType::CPU)
        .value("CUDA", DeviceType::CUDA);

    py::class_<Device>(m, "Device")
        .def(py::init<>())
        .def_readwrite("type", &Device::type)
        .def_readwrite("index", &Device::index);

    py::enum_<DType>(m, "DType")
        .value("Float32", DType::Float32)
        .value("Float16", DType::Float16)
        .value("Int32", DType::Int32)
        .value("UInt8", DType::UInt8);

    py::class_<Tensor>(m, "Tensor")
        .def_static("empty", &Tensor::empty, py::arg("shape"), py::arg("dtype"), py::arg("device") = Device())
        .def("size", &Tensor::size)
        .def("shape", &Tensor::shape)
        .def("dtype", &Tensor::dtype)
        .def("device", &Tensor::device)
        .def("to", &Tensor::to, py::arg("device"), py::arg("nonBlocking") = false)
        .def("contiguous", &Tensor::contiguous)
        .def("add", &Tensor::add)
        .def("sub", &Tensor::sub)
        .def("mul", &Tensor::mul)
        .def("div", &Tensor::div)
        .def("mm", &Tensor::mm)
        .def("transpose", &Tensor::transpose)
        .def("det", &Tensor::det)
        .def("reshape", &Tensor::reshape)
        .def("squeeze", &Tensor::squeeze)
        .def("expand", &Tensor::expand)
        .def("dataList", &Tensor::dataList)
        .def("data", [](Tensor &t) -> py::memoryview {
            size_t n = t.size();
            switch(t.dtype()) {
                case DType::Float32:
                    return py::memoryview::from_buffer(static_cast<float*>(t.data()), {n}, {sizeof(float)});
                case DType::Float16:
                    return py::memoryview::from_buffer(static_cast<uint16_t*>(t.data()), {n}, {sizeof(uint16_t)});
                case DType::Int32:
                    return py::memoryview::from_buffer(static_cast<int*>(t.data()), {n}, {sizeof(int)});
                case DType::UInt8:
                    return py::memoryview::from_buffer(static_cast<uint8_t*>(t.data()), {n}, {sizeof(uint8_t)});
                default:
                    throw std::runtime_error("Unsupported dtype for memoryview");
            }
        });
}
