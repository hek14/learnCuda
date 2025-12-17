#include "esa_utils.h"

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
 * @param queries: a list of tensors. { batch_size * [num_q_heads, dim] }
 * @param repre_cache: [N, num_k_heads, dim]
 * @param score: [S]
 * @param block_table: [S]
 * @param batch_index: [S]
 */
__global__ void retrieval_kernel_fp32(float **queries, float *__restrict__ repre_cache, float *__restrict__ score, int *__restrict__ block_table, int *__restrict__ batch_index, int num_q_heads, int num_k_heads, int dim, int S){
    if (blockIdx.x >= S){
        return;
    }
    int warp_size = 32;
    extern __shared__ float local_score[];
    float *q_offset = queries[batch_index[blockIdx.x]];
    float *k_offset = repre_cache + block_table[blockIdx.x] * num_k_heads * dim;
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
                float q_val = *(q_offset + q_head * dim + d);
                float k_val = *(k_offset + k_head * dim + d);
                sum += q_val * k_val;
                // printf("batch_id: %d, block_id: %d, q_head_id: %d, k_head_id: %d, d: %d, q_val: %f, k_val: %f\n",batch_index[blockIdx.x], block_table[blockIdx.x], q_head, k_head, d, q_val, k_val);
            }
        }
    }
    // printf("sum: %f, %d %d %d\n", sum, block_table[blockIdx.x], threadIdx.y, threadIdx.x);

    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int numWarps = ceildiv(blockDim.x * blockDim.y, warp_size);
    int warp_id = tid / numWarps;
    int lane_id = tid & (warp_size - 1);

    float warp_sum = warpReduceSum(sum);
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

__global__ void retrieval_kernel_fp16(__half **queries, __half *__restrict__ repre_cache, __half *__restrict__ score, int *__restrict__ block_table, int *__restrict__ batch_index, int dim, int S){
    extern __shared__ float local_score_fp16[]; // num of threads
    int global_x = blockIdx.x;
    int local_x = threadIdx.x;
    if (global_x < S){
        const __half *q = queries[batch_index[global_x]];
        const __half *k = repre_cache + block_table[global_x] * dim;
        int num_tiles = (dim + 2 * blockDim.x - 1) / (2 * blockDim.x);
        float sum = 0.0f;
        for(int i = 0; i < num_tiles; ++i){
            int tile_offset = i * (2 * blockDim.x);
            int idx = tile_offset + local_x * 2;
            if(idx + 2 <= dim){
                __half2 q2 = *reinterpret_cast<const __half2*>(q + idx);
                __half2 k2 = *reinterpret_cast<const __half2*>(k + idx);
                __half2 p = __hmul2(q2, k2);
                sum += __half2float(p.x) + __half2float(p.y);
            }
        }
        local_score_fp16[local_x] = sum;
        __syncthreads();
        for(int i = blockDim.x / 2; i; i = i / 2){
            if(local_x < i){
                local_score_fp16[local_x] += local_score_fp16[local_x + i];
            }
            __syncthreads();
        }
        if (local_x == 0) score[global_x] = __float2half(local_score_fp16[0]);
    }
}

__global__ void retrieval_kernel_bf16(__nv_bfloat16** queries, __nv_bfloat16* __restrict__ repre_cache, __nv_bfloat16*  __restrict__ score, int* __restrict__ block_table, int* __restrict__ batch_index, int dim, int S){
    extern __shared__ float local_score_bf16[];
    int global_x = blockIdx.x;
    int local_x  = threadIdx.x;
    if (global_x >= S) return;
    const __nv_bfloat16* q = queries[batch_index[global_x]];
    const __nv_bfloat16* k = repre_cache + block_table[global_x] * dim;
    int num_tiles = (dim + 2 * blockDim.x - 1) / (2 * blockDim.x);
    float sum = 0.0f;
    for (int i = 0; i < num_tiles; ++i) {
        int idx = i * (2 * blockDim.x) + local_x * 2;
        if (idx + 2 <= dim) {
            uint4 tmp   = *reinterpret_cast<const uint4*>(q + idx);
            uint2 q2u   = make_uint2(tmp.x, tmp.y);      // 前 4 个 bf16
            tmp         = *reinterpret_cast<const uint4*>(k + idx);
            uint2 k2u   = make_uint2(tmp.x, tmp.y);
            __nv_bfloat162 q2, k2;
            asm volatile("mov.b32 {%0, %1}, %2;"
                    : "=h"(*reinterpret_cast<uint16_t*>(&q2.x)),
                    "=h"(*reinterpret_cast<uint16_t*>(&q2.y))
                    : "r"(q2u.x));
            asm volatile("mov.b32 {%0, %1}, %2;"
                    : "=h"(*reinterpret_cast<uint16_t*>(&k2.x)),
                    "=h"(*reinterpret_cast<uint16_t*>(&k2.y))
                    : "r"(k2u.x));
            __nv_bfloat162 p = __hmul2(q2, k2);
            sum += __bfloat162float(p.x) + __bfloat162float(p.y);
        }
    }
    local_score_bf16[local_x] = sum;
    __syncthreads();
    for (int i = blockDim.x / 2; i > 0; i >>= 1) {
        if (local_x < i)
            local_score_bf16[local_x] += local_score_bf16[local_x + i];
        __syncthreads();
    }
    if (local_x == 0) score[global_x] = __float2bfloat16(local_score_bf16[0]);
}
