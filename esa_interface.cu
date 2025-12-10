#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <vector_types.h>
#include <cub/cub.cuh>
#include <torch/extension.h>
#include <vector>
#include <torch/types.h>

#define STRINGFY(func) #func
#define TORCH_BINDING_COMMON_EXTENSION(func) \
    m.def(STRINGFY(func), &func, STRINGFY(func));
#define CHECK_TORCH_TENSOR_DTYPE(T, expect_type) \
    if (((T).options().dtype() != (expect_type))) { \
        std::cout << "Got input tensor: " << (T).options() << std::endl; \
        std::cout <<"But the kernel should accept tensor with " << (expect_type) << " dtype" << std::endl; \
        throw std::runtime_error("mismatched tensor dtype"); \
    }
#define cuda_check(call){ \
    cudaError_t err = call; \
    if(err != cudaSuccess){ \
        fprintf(stderr, "cuda_error %s %d %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
    } \
} \

/**
 * This kernel performs: repre_cache[repre_block_table[i]] = mean( key_cache[key_block_table[i]], 0 )
 *
 * @param key_cache: [N, block_size, dim]
 * @param repre_cache: [N, dim]
 * @param key_block_table: [S]
 * @param repre_block_table: [S]
 */
template <typename scalar_t>
__global__ void extract_repre(const scalar_t *key_cache, scalar_t *repre_cache, const int *key_block_table, const int *repre_block_table, int block_size, int dim) {
    int idx = blockIdx.x;
    int block_id = key_block_table[idx];
    int block_id_2 = repre_block_table[idx];
    const scalar_t* key_ptr = key_cache + block_id * block_size * dim;
    scalar_t* repre_ptr = repre_cache + block_id_2 * dim;
    int d = threadIdx.x;
    if (d < dim) {
        float sum = 0;
        for (int j = 0; j < block_size; ++j) {
            sum += static_cast<float>(key_ptr[j * dim + d]);
        }
        repre_ptr[d] = static_cast<scalar_t>(sum / block_size);
    }
}

/**
 * This kernel performs: score[i] = queries[batch_index[i]] * repre_cache[block_table[i]]
 *
 * @param queries: a list of tensors. { [dim] }
 * @param repre_cache: [N, dim]
 * @param score: [S]
 * @param block_table: [S]
 * @param batch_index: [S]
 */
__global__ void retrieval_kernel(float **queries, const float *__restrict__ repre_cache, float *__restrict__ score, const int *__restrict__ block_table, const int *__restrict__ batch_index, int dim, int S){
    extern __shared__ float local_score[]; // num of threads
    int global_x = blockIdx.x;
    int local_x = threadIdx.x;
    if (global_x < S){
        const float *q = queries[batch_index[global_x]];
        const float *k = repre_cache + block_table[global_x] * dim;
        int num_tiles = (dim + 4 * blockDim.x - 1) / (4 * blockDim.x);
        float sum = 0.0f;
        for(int i = 0; i < num_tiles; ++i){
            int tile_offset = i * (4 * blockDim.x);
            int idx = tile_offset + local_x * 4;
            if(idx + 4 <= dim){
                const float4 *q4 = reinterpret_cast<const float4*>(q + idx);
                const float4 *k4 = reinterpret_cast<const float4*>(k + idx);
                sum += q4->x * k4->x + q4->y * k4->y + q4->z * k4->z + q4->w * k4->w;
            }
        }
        local_score[local_x] = sum;
        __syncthreads();
        for(int i = blockDim.x / 2; i; i = i / 2){
            if(local_x < i){
                local_score[local_x] = local_score[local_x] + local_score[local_x + i];
            }
            __syncthreads();
        }
        score[global_x] = local_score[0];
    }
}

__global__ void retrieval_kernel_fp16(half **queries, const half *__restrict__ repre_cache, half *__restrict__ score, const int *__restrict__ block_table, const int *__restrict__ batch_index, int dim, int S){
    extern __shared__ half local_score[]; // num of threads
    int global_x = blockIdx.x;
    int local_x = threadIdx.x;
    if (global_x < S){
        const half *q = queries[batch_index[global_x]];
        const half *k = repre_cache + block_table[global_x] * dim;
        int num_tiles = (dim + 4 * blockDim.x - 1) / (4 * blockDim.x);
        half sum = 0.0f;
        for(int i = 0; i < num_tiles; ++i){
            int tile_offset = i * (4 * blockDim.x);
            int idx = tile_offset + local_x * 4;
            if(idx + 4 <= dim){
                const __half4 *q4 = reinterpret_cast<const __half4*>(q + idx);
                const __half4 *k4 = reinterpret_cast<const __half4*>(k + idx);
                sum += q4->x * k4->x + q4->y * k4->y + q4->z * k4->z + q4->w * k4->w;
            }
        }
        local_score[local_x] = sum;
        __syncthreads();
        for(int i = blockDim.x / 2; i; i = i / 2){
            if(local_x < i){
                local_score[local_x] = local_score[local_x] + local_score[local_x + i];
            }
            __syncthreads();
        }
        score[global_x] = local_score[0];
    }
}

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

__global__ void retrieval_kernel_bf16(const __nv_bfloat16* bf16_tensor, float* float_tensor, size_t size) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    // We process 4 bfloat16 elements per thread in this example (8 bytes total)
    // Adjust logic if you want to use full float4 (16 bytes) loads for 8 bf16s

    if (tid * 4 + 3 < size) {
        // 1. Efficiently load two pairs of bfloat16 into __nv_bfloat162 structs.
        // Reinterpret the pointer to enable a single 32-bit load for two bfloat16s.
        const __nv_bfloat162* b162_ptr = reinterpret_cast<const __nv_bfloat162*>(bf16_tensor + tid * 4);

        __nv_bfloat162 bf16_pair1 = b162_ptr[0];
        __nv_bfloat162 bf16_pair2 = b162_ptr[1];

        // 2. Use the intrinsic to convert each pair to float2.
        float2 f32_pair1 = __bfloat1622float2(bf16_pair1);
        float2 f32_pair2 = __bfloat1622float2(bf16_pair2);

        // 3. Store the resulting floats (or use them in computation).
        // If you need a float4, you can construct it like this:
        // float4 result_f4 = make_float4(f32_pair1.x, f32_pair1.y, f32_pair2.x, f32_pair2.y);

        // Store to output tensor
        float_tensor[tid * 4] = f32_pair1.x * 2;
        float_tensor[tid * 4 + 1] = f32_pair1.y * 2;
        float_tensor[tid * 4 + 2] = f32_pair2.x * 2;
        float_tensor[tid * 4 + 3] = f32_pair2.y * 2;
    }
}

void simple_test(torch::Tensor input, torch::Tensor output){
    size_t length = input.size(0);
    size_t numThreads = 32;
    size_t numBlocks = ((length + numThreads - 1) / numThreads);
    // NOTE: Do not instantiate ptr directly by "data_ptr<__nv_bfloat16>()"
    // This will cause " undefined symbol: at10TensorBase8data_ptrI13__nv_bfloat16"
    auto *input_ptr = reinterpret_cast<const __nv_bfloat16*>(input.data_ptr());
    auto *output_ptr = reinterpret_cast<float*>(output.data_ptr());
    retrieval_kernel_bf16<<<numBlocks, numThreads>>>(input_ptr, output_ptr, length);
}


void esa_retrieval(const std::vector<torch::Tensor> &query_list, torch::Tensor repre_cache, torch::Tensor q_index, torch::Tensor repre_index, torch::Tensor score, torch::Tensor score_sorted, torch::Tensor index_ranged, torch::Tensor index_sorted, torch::Tensor batch_offset, torch::Tensor workspace){
    int s = q_index.size(0);
    int dim = repre_cache.size(1);
    int batch = query_list.size();
    dim3 numThreads = {(unsigned int)(32)};
    dim3 numBlocks = {(unsigned int) s};
    size_t bytes = numThreads.x * sizeof(float);


    // method 1: use cudaMallocManaged to allocate unified_memory, this perform really good
    float** Q_ptrs = nullptr;
    cudaMallocManaged(&Q_ptrs, batch * sizeof(float*));
    for(int i = 0; i < batch; ++i) {
        Q_ptrs[i] = query_list[i].data_ptr<float>();
    }

    // method 2: copy pointers from host to device, this will spent more time
    // std::vector<float*> h_Q_ptrs(batch);
    // for(int i = 0; i < batch; ++i) {
    //     h_Q_ptrs[i] = query_list[i].data_ptr<float>();
    // }
    // float **Q_ptrs;
    // cuda_check(cudaMalloc(&Q_ptrs, batch * sizeof(float*)));
    // cuda_check(cudaMemcpy(Q_ptrs, h_Q_ptrs.data(), batch * sizeof(float*), cudaMemcpyHostToDevice));

    retrieval_kernel<<<numBlocks, numThreads, bytes>>>(Q_ptrs, repre_cache.data_ptr<float>(), score.data_ptr<float>(), repre_index.data_ptr<int>(), q_index.data_ptr<int>(), dim, s);

    void* temp_workspace = nullptr;
    size_t temp_bytes = 0;
    cub::DeviceSegmentedRadixSort::SortPairsDescending(
        temp_workspace, temp_bytes,
        score.data_ptr<float>(),  score_sorted.data_ptr<float>(),
        index_ranged.data_ptr<int>(), index_sorted.data_ptr<int>(),
        s, batch, batch_offset.data_ptr<int>(), batch_offset.data_ptr<int>() + 1);
    temp_workspace = workspace.data_ptr<int>();
    cub::DeviceSegmentedRadixSort::SortPairsDescending(
        temp_workspace, temp_bytes,
        score.data_ptr<float>(),  score_sorted.data_ptr<float>(),
        index_ranged.data_ptr<int>(), index_sorted.data_ptr<int>(),
        s, batch, batch_offset.data_ptr<int>(), batch_offset.data_ptr<int>() + 1);
    cuda_check(cudaFree(Q_ptrs));
}

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
    // NOTE: don't malloc, just reuse the workspace, but the first call of
    // SortPairsDescending is necesssary to determine the workspace size
    // cuda_check(cudaMalloc(&temp_workspace, temp_bytes));
    temp_workspace = workspace.data_ptr<int>();

    cub::DeviceSegmentedRadixSort::SortPairsDescending(
        temp_workspace, temp_bytes,
        score.data_ptr<float>(),  score_out.data_ptr<float>(),
        index.data_ptr<int>(), index_out.data_ptr<int>(),
        total, B, offsets.data_ptr<int>(), offsets.data_ptr<int>() + 1);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    TORCH_BINDING_COMMON_EXTENSION(esa_retrieval)
    TORCH_BINDING_COMMON_EXTENSION(esa_topk)
    TORCH_BINDING_COMMON_EXTENSION(esa_repre)
    TORCH_BINDING_COMMON_EXTENSION(simple_test)
}
