#include <cub/cub.cuh>
#include <cstddef>
#include <torch/extension.h>

__inline__ __device__ float warpReduceSum(float val)
{
    int warpSize = 32;
    unsigned mask = __activemask();          // ballot of *all* currently active threads
    for (int offset = warpSize/2; offset > 0; offset /= 2)
        val += __shfl_down_sync(mask, val, offset);
    return val;                              // only lane 0 holds the total
}

constexpr __host__ __device__
int ceildiv(int a, int b) {
    return (a + b - 1) / b;
}
/**
 * This kernel performs: repre_cache[repre_repre_index[i]] = mean( key_cache[key_repre_index[i]], 0 )
 *
 * @param key_cache: [N, block_size, dim]
 * @param repre_cache: [N, dim]
 * @param key_repre_index: [S]
 * @param repre_repre_index: [S]
 */
template <typename scalar_t>
__global__ void extract_repre(const scalar_t *key_cache, scalar_t *repre_cache, const int *key_repre_index, const int *repre_index, int block_size, int dim) {
    int idx = blockIdx.x;
    int block_id = key_repre_index[idx];
    int block_id_2 = repre_index[idx];
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
 * This kernel performs: score[i] = queries[query_index[i]] * repre_cache[repre_index[i]]
 *
 * @param queries: a list of tensors. { batch_size * [num_q_heads, dim] }
 * @param repre_cache: [N, num_k_heads, dim]
 * @param score: [S]
 * @param repre_index: [S]
 * @param query_index: [S]
 */

__global__ void retrieval_kernel_fp16(__half *__restrict__ queries, __half *__restrict__ repre_cache, __half *__restrict__ score, int *__restrict__ repre_index, int *__restrict__ query_index, int num_q_heads, int num_k_heads, int dim, int S){
    if (blockIdx.x >= S){
        return;
    }
    int warp_size = 32;
    extern __shared__ float local_score[];
    auto *q_offset = queries + query_index[blockIdx.x] * num_q_heads * dim;
    auto *k_offset = repre_cache + repre_index[blockIdx.x] * num_k_heads * dim;
    int num_tiles_y = ceildiv(num_q_heads, blockDim.y);
    int num_tiles_x = ceildiv(dim, blockDim.x);
    int gqa_size = num_q_heads / num_k_heads;

    float sum = 0.0f;
    for (int y = 0; y < num_tiles_y; ++y){
        int q_head = y * blockDim.y + threadIdx.y;
        int k_head = q_head / gqa_size;
        for(int x = 0; x < num_tiles_x; ++x){
            int d = x * blockDim.x + threadIdx.x;
            if (q_head < num_q_heads && k_head < num_k_heads && d < dim){
                auto q_val = *(q_offset + q_head * dim + d);
                auto k_val = *(k_offset + k_head * dim + d);
                sum += __half2float(q_val) * __half2float(k_val);
            }
        }
    }

    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int numWarps = ceildiv(blockDim.x * blockDim.y, warp_size);
    int warp_id = tid / numWarps;
    int lane_id = tid & (warp_size - 1);

    auto warp_sum = warpReduceSum(sum);
    if(lane_id == 0){
        local_score[warp_id] = warp_sum;
    }
    __syncthreads();
    if(warp_id == 0){
        sum = lane_id < numWarps ? local_score[lane_id] : 0.0f;
        sum = warpReduceSum(sum);
        if(lane_id == 0){
            score[blockIdx.x] = __float2half(sum);
        }
    }
}


__global__ void retrieval_kernel_fp32(float *__restrict__ queries, float *__restrict__ repre_cache, float *__restrict__ score, int *__restrict__ repre_index, int *__restrict__ query_index, int num_q_heads, int num_k_heads, int dim, int S){
    if (blockIdx.x >= S){
        return;
    }
    int warp_size = 32;
    extern __shared__ float local_score[];
    auto *q_offset = queries + query_index[blockIdx.x] * num_q_heads * dim;
    auto *k_offset = repre_cache + repre_index[blockIdx.x] * num_k_heads * dim;
    int num_tiles_y = ceildiv(num_q_heads, blockDim.y);
    int num_tiles_x = ceildiv(dim, blockDim.x);
    int gqa_size = num_q_heads / num_k_heads;

    float sum = 0.0f;
    for (int y = 0; y < num_tiles_y; ++y){
        int q_head = y * blockDim.y + threadIdx.y;
        int k_head = q_head / gqa_size;
        for(int x = 0; x < num_tiles_x; ++x){
            int d = x * blockDim.x + threadIdx.x;
            if (q_head < num_q_heads && k_head < num_k_heads && d < dim){
                auto q_val = *(q_offset + q_head * dim + d);
                auto k_val = *(k_offset + k_head * dim + d);
                sum += q_val * k_val;
            }
        }
    }

    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int numWarps = ceildiv(blockDim.x * blockDim.y, warp_size);
    int warp_id = tid / numWarps;
    int lane_id = tid & (warp_size - 1);

    auto warp_sum = warpReduceSum(sum);
    if(lane_id == 0){
        local_score[warp_id] = warp_sum;
    }
    __syncthreads();
    if(warp_id == 0){
        sum = lane_id < numWarps ? local_score[lane_id] : 0.0f;
        sum = warpReduceSum(sum);
        if(lane_id == 0){
            score[blockIdx.x] = sum;
        }
    }
}

__global__ void retrieval_kernel_bf16(__nv_bfloat16 *__restrict__ queries, __nv_bfloat16 *__restrict__ repre_cache, __nv_bfloat16 *__restrict__ score, int *__restrict__ repre_index, int *__restrict__ query_index, int num_q_heads, int num_k_heads, int dim, int S){
    if (blockIdx.x >= S){
        return;
    }
    int warp_size = 32;
    extern __shared__ float local_score[];
    auto *q_offset = queries + query_index[blockIdx.x] * num_q_heads * dim;
    auto *k_offset = repre_cache + repre_index[blockIdx.x] * num_k_heads * dim;
    int num_tiles_y = ceildiv(num_q_heads, blockDim.y);
    int num_tiles_x = ceildiv(dim, blockDim.x);
    int gqa_size = num_q_heads / num_k_heads;

    float sum = 0.0f;
    for (int y = 0; y < num_tiles_y; ++y){
        int q_head = y * blockDim.y + threadIdx.y;
        int k_head = q_head / gqa_size;
        for(int x = 0; x < num_tiles_x; ++x){
            int d = x * blockDim.x + threadIdx.x;
            if (q_head < num_q_heads && k_head < num_k_heads && d < dim){
                auto q_val = *(q_offset + q_head * dim + d);
                auto k_val = *(k_offset + k_head * dim + d);
                sum += __bfloat162float(q_val) * __bfloat162float(k_val);
            }
        }
    }

    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int numWarps = ceildiv(blockDim.x * blockDim.y, warp_size);
    int warp_id = tid / numWarps;
    int lane_id = tid & (warp_size - 1);

    auto warp_sum = warpReduceSum(sum);
    if(lane_id == 0){
        local_score[warp_id] = warp_sum;
    }
    __syncthreads();
    if(warp_id == 0){
        sum = lane_id < numWarps ? local_score[lane_id] : 0.0f;
        sum = warpReduceSum(sum);
        if(lane_id == 0){
            score[blockIdx.x] = __float2bfloat16(sum);
        }
    }
}


extern "C" void esa_repre(torch::Tensor key_cache, torch::Tensor repre_cache, torch::Tensor block_table, torch::Tensor repre_table){
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

extern "C" void esa_topk(torch::Tensor score, torch::Tensor index, torch::Tensor offsets, torch::Tensor score_out, torch::Tensor index_out, torch::Tensor workspace){
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


extern "C" void esa_retrieval_launcher(torch::Tensor query, torch::Tensor repre_cache, torch::Tensor q_index, torch::Tensor repre_index, torch::Tensor batch_offset, torch::Tensor workspace, torch::Tensor score, torch::Tensor score_sorted, torch::Tensor index, torch::Tensor index_sorted){
    int s = repre_index.size(0);
    int num_k_heads = repre_cache.size(1);
    int num_q_heads = query.size(1);
    int dim = repre_cache.size(2);
    int batch = batch_offset.size(0) - 1;

    dim3 numBlocks = {(unsigned int)(s)};
    dim3 numThreads = {32, 32};
    int numWarps = ceildiv(numThreads.x * numThreads.y, 32);
    size_t bytes = numWarps * sizeof(float);
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, repre_cache.scalar_type(), "esa_retrieval_cuda", ([&] {
                if constexpr (std::is_same_v<scalar_t, float>) {
                retrieval_kernel_fp32<<<numBlocks, numThreads, bytes>>>(reinterpret_cast<float*>(query.data_ptr()),
                        reinterpret_cast<float*>(repre_cache.data_ptr()), reinterpret_cast<float*>(score.data_ptr()), repre_index.data_ptr<int>(), q_index.data_ptr<int>(), num_q_heads, num_k_heads, dim, s);
                void* temp_workspace = nullptr;
                size_t temp_bytes = 0;
                cub::DeviceSegmentedRadixSort::SortPairsDescending(
                        temp_workspace, temp_bytes,
                        score.data_ptr<float>(),  score_sorted.data_ptr<float>(),
                        index.data_ptr<int>(), index_sorted.data_ptr<int>(),
                        s, batch, batch_offset.data_ptr<int>(), batch_offset.data_ptr<int>() + 1);
                temp_workspace = workspace.data_ptr<int>();
                cub::DeviceSegmentedRadixSort::SortPairsDescending(
                        temp_workspace, temp_bytes,
                        score.data_ptr<float>(),  score_sorted.data_ptr<float>(),
                        index.data_ptr<int>(), index_sorted.data_ptr<int>(),
                        s, batch, batch_offset.data_ptr<int>(), batch_offset.data_ptr<int>() + 1);
                } else if constexpr (std::is_same_v<scalar_t, at::Half>) {
                retrieval_kernel_fp16<<<numBlocks, numThreads, bytes>>>(reinterpret_cast<__half*>(query.data_ptr()),
                        reinterpret_cast<__half*>(repre_cache.data_ptr()), reinterpret_cast<__half*>(score.data_ptr()), repre_index.data_ptr<int>(), q_index.data_ptr<int>(), num_q_heads, num_k_heads, dim, s);
                void* temp_workspace = nullptr;
                size_t temp_bytes = 0;
                cub::DeviceSegmentedRadixSort::SortPairsDescending(
                        temp_workspace, temp_bytes,
                        reinterpret_cast<__half*>(score.data_ptr()),
                        reinterpret_cast<__half*>(score_sorted.data_ptr()),
                        index.data_ptr<int>(), index_sorted.data_ptr<int>(),
                        s, batch, batch_offset.data_ptr<int>(), batch_offset.data_ptr<int>() + 1);
                temp_workspace = workspace.data_ptr<int>();
                cub::DeviceSegmentedRadixSort::SortPairsDescending(
                        temp_workspace, temp_bytes,
                        reinterpret_cast<__half*>(score.data_ptr()),
                        reinterpret_cast<__half*>(score_sorted.data_ptr()),
                        index.data_ptr<int>(), index_sorted.data_ptr<int>(),
                        s, batch, batch_offset.data_ptr<int>(), batch_offset.data_ptr<int>() + 1);
                } else if constexpr (std::is_same_v<scalar_t, at::BFloat16>) {
                    retrieval_kernel_bf16<<<numBlocks, numThreads, bytes>>>(reinterpret_cast<__nv_bfloat16*>(query.data_ptr()),
                            reinterpret_cast<__nv_bfloat16*>(repre_cache.data_ptr()), reinterpret_cast<__nv_bfloat16*>(score.data_ptr()), repre_index.data_ptr<int>(), q_index.data_ptr<int>(), num_q_heads, num_k_heads, dim, s);
                    void* temp_workspace = nullptr;
                    size_t temp_bytes = 0;
                    cub::DeviceSegmentedRadixSort::SortPairsDescending(
                            temp_workspace, temp_bytes,
                            reinterpret_cast<__nv_bfloat16*>(score.data_ptr()),
                            reinterpret_cast<__nv_bfloat16*>(score_sorted.data_ptr()),
                            index.data_ptr<int>(), index_sorted.data_ptr<int>(),
                            s, batch, batch_offset.data_ptr<int>(), batch_offset.data_ptr<int>() + 1);
                    temp_workspace = workspace.data_ptr<int>();
                    cub::DeviceSegmentedRadixSort::SortPairsDescending(
                            temp_workspace, temp_bytes,
                            reinterpret_cast<__nv_bfloat16*>(score.data_ptr()),
                            reinterpret_cast<__nv_bfloat16*>(score_sorted.data_ptr()),
                            index.data_ptr<int>(), index_sorted.data_ptr<int>(),
                            s, batch, batch_offset.data_ptr<int>(), batch_offset.data_ptr<int>() + 1);
                }
    }));
}
