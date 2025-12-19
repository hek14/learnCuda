#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include "esa_kernels.h"
#include "cuda_sm_copy.h"

namespace py = pybind11;

#define STRINGFY(func) #func
#define TORCH_BINDING_COMMON_EXTENSION(func) \
    m.def(STRINGFY(func), &func, STRINGFY(func))

PYBIND11_MODULE(esa_interface, m) {
    m.doc() = "ESA cuda kernels for block feature extraction and block retrieval";
    py::class_<RetrievalInputTensor>(m, "RetrievalInputTensor")
        .def(py::init<>())
        .def_readwrite("repre_cache", &RetrievalInputTensor::repre_cache)
        .def_readwrite("q_index", &RetrievalInputTensor::q_index)
        .def_readwrite("num_q_heads", &RetrievalInputTensor::num_q_heads)
        .def_readwrite("batch_size", &RetrievalInputTensor::batch_size)
        .def_readwrite("repre_index", &RetrievalInputTensor::repre_index)
        .def_readwrite("batch_offset", &RetrievalInputTensor::batch_offset)
        .def_readwrite("q_ptrs", &RetrievalInputTensor::q_ptrs)
        .def_readwrite("workspace", &RetrievalInputTensor::workspace);

    py::class_<RetrievalOutputTensor>(m, "RetrievalOutputTensor")
        .def(py::init<>())
        .def_readwrite("score", &RetrievalOutputTensor::score)
        .def_readwrite("score_sorted", &RetrievalOutputTensor::score_sorted)
        .def_readwrite("index_ranged", &RetrievalOutputTensor::index_ranged)
        .def_readwrite("index_sorted", &RetrievalOutputTensor::index_sorted);

    TORCH_BINDING_COMMON_EXTENSION(esa_retrieval);
    TORCH_BINDING_COMMON_EXTENSION(esa_topk);
    TORCH_BINDING_COMMON_EXTENSION(esa_repre);
    TORCH_BINDING_COMMON_EXTENSION(esa_copy);
}
