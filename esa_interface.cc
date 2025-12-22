#include <stdexcept>
#include <string>
#include <vector>

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

extern "C" void esa_retrieval_launcher(torch::Tensor q_ptrs, torch::Tensor repre_cache, torch::Tensor q_index, torch::Tensor repre_index,
        torch::Tensor batch_offset, torch::Tensor workspace, torch::Tensor score, torch::Tensor score_sorted, torch::Tensor index_ranged, torch::Tensor index_sorted, int num_q_heads, int batch_size);

struct RetrievalInputTensor{
    torch::Tensor q_ptrs;
    torch::Tensor repre_cache;
    torch::Tensor q_index;
    torch::Tensor repre_index;
    torch::Tensor batch_offset;
    torch::Tensor workspace;
    int num_q_heads;
    int batch_size;
};

struct RetrievalOutputTensor{
    torch::Tensor score;
    torch::Tensor score_sorted;
    torch::Tensor index_ranged;
    torch::Tensor index_sorted;
};


void esa_retrieval(RetrievalInputTensor input, RetrievalOutputTensor output){
    auto q_ptrs = input.q_ptrs;
    auto repre_cache = input.repre_cache;
    auto q_index = input.q_index;
    auto repre_index = input.repre_index;
    auto batch_offset = input.batch_offset;
    auto workspace = input.workspace;
    auto num_q_heads = input.num_q_heads;
    auto batch_size = input.batch_size;
    auto score = output.score;
    auto score_sorted = output.score_sorted;
    auto index_ranged = output.index_ranged;
    auto index_sorted = output.index_sorted;
    esa_retrieval_launcher(q_ptrs, repre_cache, q_index, repre_index, batch_offset, workspace, score, score_sorted, index_ranged, index_sorted, num_q_heads, batch_size);
}


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
}
