#include "esa_kernels.cu"

void esa_repre(torch::Tensor key_cache, torch::Tensor repre_cache, torch::Tensor block_table, torch::Tensor repre_table){
    int block_size = key_cache.size(1);
    int dim = repre_cache.size(-1);
    int threads = dim;
    int blocks = block_table.size(0);
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, key_cache.scalar_type(), "esa_repre_cuda", ([&] {
        extract_repre<scalar_t><<<blocks, threads>>>(
            key_cache.data_ptr<scalar_t>(),
            repre_cache.data_ptr<scalar_t>(),
            block_table.data_ptr<int>(),
            repre_table.data_ptr<int>(),
            block_size,
            dim);
        }));
}

struct RetrievalInputTensor{
    torch::Tensor repre_cache;
    torch::Tensor q_index;
    torch::Tensor repre_index;
    torch::Tensor batch_offset;
    torch::Tensor q_ptrs;
    torch::Tensor workspace;
    int num_q_heads;
};

struct RetrievalOutputTensor{
    torch::Tensor score;
    torch::Tensor score_sorted;
    torch::Tensor index_ranged;
    torch::Tensor index_sorted;
};

// void esa_retrieval(RetrievalInputTensor input, RetrievalOutputTensor output){
//     auto q_ptrs = input.q_ptrs;
//     auto repre_cache = input.repre_cache;
//     auto q_index = input.q_index;
//     auto repre_index = input.repre_index;
//     auto batch_offset = input.batch_offset;
//     auto workspace = input.workspace;
//
//     auto score = output.score;
//     auto score_sorted = output.score_sorted;
//     auto index_ranged = output.index_ranged;
//     auto index_sorted = output.index_sorted;
//
//     int s = q_index.size(0);
//     int dim = repre_cache.size(1);
//     int batch = q_ptrs.size(0);
//     dim3 numThreads = {(unsigned int)(32)};
//     dim3 numBlocks = {(unsigned int)(s)};
//
//     AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, repre_cache.scalar_type(), "esa_retrieval_cuda", [&]{
//         if constexpr (std::is_same_v<scalar_t, float>) {
//             float** Q_ptrs = reinterpret_cast<float**>(input.q_ptrs.data_ptr<int64_t>());
//             size_t bytes = numThreads.x * sizeof(float);
//             retrieval_kernel_fp32<<<numBlocks, numThreads, bytes>>>(Q_ptrs, repre_cache.data_ptr<float>(), score.data_ptr<float>(), repre_index.data_ptr<int>(), q_index.data_ptr<int>(), dim, s);
//             void* temp_workspace = nullptr;
//             size_t temp_bytes = 0;
//             cub::DeviceSegmentedRadixSort::SortPairsDescending(
//                     temp_workspace, temp_bytes,
//                     score.data_ptr<float>(),  score_sorted.data_ptr<float>(),
//                     index_ranged.data_ptr<int>(), index_sorted.data_ptr<int>(),
//                     s, batch, batch_offset.data_ptr<int>(), batch_offset.data_ptr<int>() + 1);
//             temp_workspace = workspace.data_ptr<int>();
//             cub::DeviceSegmentedRadixSort::SortPairsDescending(
//                     temp_workspace, temp_bytes,
//                     score.data_ptr<float>(),  score_sorted.data_ptr<float>(),
//                     index_ranged.data_ptr<int>(), index_sorted.data_ptr<int>(),
//                     s, batch, batch_offset.data_ptr<int>(), batch_offset.data_ptr<int>() + 1);
//         } else if constexpr (std::is_same_v<scalar_t, at::Half>) {
//             __half** Q_ptrs = reinterpret_cast<__half**>(input.q_ptrs.data_ptr<int64_t>());
//             size_t bytes = numThreads.x * sizeof(float);
//             retrieval_kernel_fp16<<<numBlocks, numThreads, bytes>>>(Q_ptrs,
//                     reinterpret_cast<__half*>(repre_cache.data_ptr()),
//                     reinterpret_cast<__half*>(score.data_ptr()),
//                     reinterpret_cast<int*>(repre_index.data_ptr()),
//                     reinterpret_cast<int*>(q_index.data_ptr()),
//                     dim, s);
//             void* temp_workspace = nullptr;
//             size_t temp_bytes = 0;
//             cub::DeviceSegmentedRadixSort::SortPairsDescending(
//                     temp_workspace, temp_bytes,
//                     reinterpret_cast<__half*>(score.data_ptr()),
//                     reinterpret_cast<__half*>(score_sorted.data_ptr()),
//                     index_ranged.data_ptr<int>(), index_sorted.data_ptr<int>(),
//                     s, batch, batch_offset.data_ptr<int>(), batch_offset.data_ptr<int>() + 1);
//             temp_workspace = workspace.data_ptr<int>();
//             cub::DeviceSegmentedRadixSort::SortPairsDescending(
//                     temp_workspace, temp_bytes,
//                     reinterpret_cast<__half*>(score.data_ptr()),
//                     reinterpret_cast<__half*>(score_sorted.data_ptr()),
//                     index_ranged.data_ptr<int>(), index_sorted.data_ptr<int>(),
//                     s, batch, batch_offset.data_ptr<int>(), batch_offset.data_ptr<int>() + 1);
//         } else if constexpr (std::is_same_v<scalar_t, at::BFloat16>) {
//             __nv_bfloat16** Q_ptrs = reinterpret_cast<__nv_bfloat16**>(input.q_ptrs.data_ptr<int64_t>());
//             size_t bytes = numThreads.x * sizeof(float);
//             retrieval_kernel_bf16<<<numBlocks, numThreads, bytes>>>(Q_ptrs,
//                     reinterpret_cast<__nv_bfloat16*>(repre_cache.data_ptr()),
//                     reinterpret_cast<__nv_bfloat16*>(score.data_ptr()),
//                     reinterpret_cast<int*>(repre_index.data_ptr()),
//                     reinterpret_cast<int*>(q_index.data_ptr()),
//                     dim, s);
//             void* temp_workspace = nullptr;
//             size_t temp_bytes = 0;
//             cub::DeviceSegmentedRadixSort::SortPairsDescending(
//                     temp_workspace, temp_bytes,
//                     reinterpret_cast<__nv_bfloat16*>(score.data_ptr()),
//                     reinterpret_cast<__nv_bfloat16*>(score_sorted.data_ptr()),
//                     index_ranged.data_ptr<int>(), index_sorted.data_ptr<int>(),
//                     s, batch, batch_offset.data_ptr<int>(), batch_offset.data_ptr<int>() + 1);
//             temp_workspace = workspace.data_ptr<int>();
//             cub::DeviceSegmentedRadixSort::SortPairsDescending(
//                     temp_workspace, temp_bytes,
//                     reinterpret_cast<__nv_bfloat16*>(score.data_ptr()),
//                     reinterpret_cast<__nv_bfloat16*>(score_sorted.data_ptr()),
//                     index_ranged.data_ptr<int>(), index_sorted.data_ptr<int>(),
//                     s, batch, batch_offset.data_ptr<int>(), batch_offset.data_ptr<int>() + 1);
//         }
//     });
// }

void esa_topk(torch::Tensor score, torch::Tensor index, torch::Tensor offsets, torch::Tensor score_out, torch::Tensor index_out, torch::Tensor workspace){
    void* temp_workspace = nullptr;
    size_t temp_bytes = 0;
    size_t B = offsets.size(0) - 1;
    size_t total = score.size(0);
    cub::DeviceSegmentedRadixSort::SortPairsDescending(
            temp_workspace, temp_bytes,
            score.data_ptr<float>(),  score_out.data_ptr<float>(),
            index.data_ptr<int>(), index_out.data_ptr<int>(),
            total, B, offsets.data_ptr<int>(), offsets.data_ptr<int>() + 1);
    // NOTE: Don't use malloc, just reuse the workspace, but the first call of
    // SortPairsDescending is necesssary to determine the workspace size.
    // CUDA_CHECK(cudaMalloc(&temp_workspace, temp_bytes));
    temp_workspace = workspace.data_ptr<int>();
    cub::DeviceSegmentedRadixSort::SortPairsDescending(
            temp_workspace, temp_bytes,
            score.data_ptr<float>(),  score_out.data_ptr<float>(),
            index.data_ptr<int>(), index_out.data_ptr<int>(),
            total, B, offsets.data_ptr<int>(), offsets.data_ptr<int>() + 1);
}

void esa_retrieval(RetrievalInputTensor input, RetrievalOutputTensor output){
    auto q_ptrs = input.q_ptrs;
    auto repre_cache = input.repre_cache;
    auto q_index = input.q_index;
    auto repre_index = input.repre_index;
    auto num_q_heads = input.num_q_heads;

    auto score = output.score;

    int s = repre_index.size(0);
    auto num_k_heads = repre_cache.size(1);
    int dim = repre_cache.size(2);
    int batch = q_ptrs.size(0);

    dim3 numBlocks = {(unsigned int)s};
    dim3 numThreads = {32, 32};
    int numWarps = ceildiv(numThreads.x * numThreads.y, 32);
    size_t bytes = numWarps * sizeof(float);
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, repre_cache.scalar_type(), "esa_retrieval_cuda", ([&] {
        auto* Q_ptrs = reinterpret_cast<scalar_t**>(q_ptrs.data_ptr<int64_t>());
        retrieval_kernel_fp32<<<numBlocks, numThreads, bytes>>>(Q_ptrs, repre_cache.data_ptr<scalar_t>(), score.data_ptr<scalar_t>(), repre_index.data_ptr<int>(), q_index.data_ptr<int>(), num_q_heads, num_k_heads, dim, s);
    }));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "ESA cuda kernels for block feature extraction and block retrieval";
    py::class_<RetrievalInputTensor>(m, "RetrievalInputTensor")
        .def(py::init<>())
        .def_readwrite("repre_cache", &RetrievalInputTensor::repre_cache)
        .def_readwrite("q_index", &RetrievalInputTensor::q_index)
        .def_readwrite("num_q_heads", &RetrievalInputTensor::num_q_heads)
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
}
