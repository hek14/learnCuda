#include <cuda_runtime.h>
#include <stdio.h>
#include <chrono>
#include <random>

#include <torch/extension.h>
#include <vector>
#include <torch/types.h>

#define cuda_check(call){ \
    cudaError_t err = call; \
    if(err != cudaSuccess){ \
        fprintf(stderr, "cuda_error %s %d %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
    } \
} \

// TODO: change Q from float* to float**
__global__ void retrieval_kernel(const float *__restrict__ Q, const float *__restrict__ K, float *__restrict__ score, const int *__restrict__ block_table, const int *__restrict__ batch_index, int dim, int S){
    // Q: [batch, dim], the query tensors
    // K: [N, dim], the key tensors
    // score: [S], the result score values
    // block_table: [S], the index for K. It's a flattened tensors which actually compose `batch` segment: [N1, N2, N3] for batch = 3, N1 + N2 + N3 = S
    // batch_index: [S], the mark specifying which batch current index belongs to and which Q current K[index] should be compared with.
    // dim: feature size
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < S){
        int k_index = block_table[idx];
        int batch_id = batch_index[idx];
        const float* pQ = Q + batch_id * dim;
        const float* pK = K + k_index * dim;
        float s = 0.0f;
        #pragma unroll 8
        for(int i = 0; i < dim; ++i){
            s += pQ[i] * pK[i];
        }
        score[idx] = s;
    }
}

#define tile 8
__global__ void retrieval_kernel_2(const float *__restrict__ Q[], const float *__restrict__ K, float *__restrict__ score, const int *__restrict__ block_table, const int *__restrict__ batch_index, int dim, int S){
    // Q: [batch, dim], the query tensors
    // K: [N, dim], the key tensors
    // score: [S], the result score values
    // block_table: [S], the index for K. It's a flattened tensors which actually compose `batch` segment: [N1, N2, N3] for batch = 3, N1 + N2 + N3 = S
    // batch_index: [S], the mark specifying which batch current index belongs to and which Q current K[index] should be compared with.
    // dim: feature size
    extern __shared__ unsigned char q_and_key[]; // dynamic size
    float *query = reinterpret_cast<float*>(q_and_key);
    float *feature = reinterpret_cast<float*>(q_and_key + sizeof(float) * dim);
    float *part_score = reinterpret_cast<float*>(q_and_key + sizeof(float) * dim * 2); // dim / tile
    int global_x = blockIdx.x;
    if (global_x < S){
        int local_x = threadIdx.x;
        const float *k = K + block_table[global_x] * dim;
        const float *q = Q[batch_index[global_x]];
        int tile_offset = local_x * tile;
        for(int i = 0; i < tile; ++i){
            if(i + tile_offset < dim){
                query[i + tile_offset] = q[i + tile_offset];
                feature[i + tile_offset] = k[i + tile_offset];
            }
        }
        __syncthreads();
        float sum = 0.0f;
        for(int i = 0; i < tile; ++i){
            if(tile_offset + i < dim){
                sum += feature[tile_offset + i] * query[tile_offset + i];
            }
        }
        part_score[local_x] = sum;
        __syncthreads();

        for(int i = blockDim.x / 2; i; i /= 2){
            if(local_x < i){
                part_score[local_x] += part_score[local_x + i];
            }
            __syncthreads();
        }
        score[global_x] = part_score[0];
    }
}

__global__ void retrieval_kernel_3(const float *__restrict__ Q[], const float *__restrict__ K, float *__restrict__ score, const int *__restrict__ block_table, const int *__restrict__ batch_index, int dim, int S){
    // Q: [batch, dim], the query tensors
    // K: [N, dim], the key tensors
    // score: [S], the result score values
    // block_table: [S], the index for K. It's a flattened tensors which actually compose `batch` segment: [N1, N2, N3] for batch = 3, N1 + N2 + N3 = S
    // batch_index: [S], the mark specifying which batch current index belongs to and which Q current K[index] should be compared with.
    // dim: feature size
    extern __shared__ float local_score[]; // num of threads
    int global_x = blockIdx.x;
    int local_x = threadIdx.x;
    if (global_x < S){
        const float *q = Q[batch_index[global_x]];
        const float *k = K + block_table[global_x] * dim;
        int num_tiles = (dim + 4 * blockDim.x - 1) / (4 * blockDim.x);
        float sum = 0.0f;
        for(int i = 0; i < num_tiles; ++i){
            int tile_offset = i * (4 * blockDim.x);
            if(tile_offset + local_x * 4 + 4 <= dim){
                const float4 *q4 = reinterpret_cast<const float4*>(q + tile_offset + local_x * 4);
                const float4 *k4 = reinterpret_cast<const float4*>(k + tile_offset + local_x * 4);
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


void retrieval_host(float *Q, float *K, float *score, int *block_table, int *batch_index, int dim, int S){
    for(int i = 0; i < S; ++i){
        int batch_id = batch_index[i];
        int k_index = block_table[i];
        float sum = 0.0f;
        for(int j = 0; j < dim; ++j){
            sum += Q[batch_id * dim + j] * K[k_index * dim + j];
        }
        score[i] = sum;
    }
}

void init_mat(float *mat, int sz){
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(0.0f, 5.0f);
    for(int i = 0; i < sz; ++i){
        mat[i] = dist(rng);
    }

}

#define STRINGFY(func) #func
#define TORCH_BINDING_COMMON_EXTENSION(func) \
    m.def(STRINGFY(func), &func, STRINGFY(func));
#define CHECK_TORCH_TENSOR_DTYPE(T, expect_type) \
if (((T).options().dtype() != (expect_type))) { \
    std::cout << "Got input tensor: " << (T).options() << std::endl; \
    std::cout <<"But the kernel should accept tensor with " << (expect_type) << " dtype" << std::endl; \
    throw std::runtime_error("mismatched tensor dtype"); \
}

void cuda_retrieval(const std::vector<torch::Tensor> &query_list, torch::Tensor repre_cache, torch::Tensor q_index, torch::Tensor repre_index, torch::Tensor score){
    // query: a list of ptr
    // repre_cache: a ptr
    CHECK_TORCH_TENSOR_DTYPE(query_list[0], torch::kFloat32);
    CHECK_TORCH_TENSOR_DTYPE(repre_cache, torch::kFloat32);
    int s = q_index.size(0);
    int dim = repre_cache.size(1);
    dim3 numThreads = {(unsigned int)(32)};
    dim3 numBlocks = {(unsigned int) s};
    size_t bytes = numThreads.x * sizeof(float);

    // build array of Q pointers
    int batch = query_list.size();
    std::vector<float*> h_Q_ptrs(batch);
    for(int i = 0; i < batch; ++i) {
        h_Q_ptrs[i] = query_list[i].data_ptr<float>();
    }
    float **d_Q_ptrs;
    cuda_check(cudaMalloc(&d_Q_ptrs, batch * sizeof(float*)));
    cuda_check(cudaMemcpy(d_Q_ptrs, h_Q_ptrs.data(), batch * sizeof(float*), cudaMemcpyHostToDevice));
    retrieval_kernel_3<<<numBlocks, numThreads, bytes>>>(d_Q_ptrs, repre_cache.data_ptr<float>(), score.data_ptr<float>(), repre_index.data_ptr<int>(), q_index.data_ptr<int>(), dim, s);
    cuda_check(cudaFree(d_Q_ptrs));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    TORCH_BINDING_COMMON_EXTENSION(cuda_retrieval)
}

// int main(){
//     float *h_Q, *h_K;
//     int B;
//     int seq_len;
//     int dim;
//     scanf("%d%d%d", &B, &seq_len, &dim);
//
//
//     int N = 4000;
//     h_Q = (float*)malloc(B * dim * sizeof(float));
//     h_K = (float*)malloc(N * dim * sizeof(float));
//
//     init_mat(h_Q, B * dim);
//     init_mat(h_K, N * dim);
//
//     int total_kv_len = 0;
//     int *h_kv_len;
//     h_kv_len = (int*)malloc(B * sizeof(int));
//     int *kv_start_offsets ;
//     kv_start_offsets = (int*)malloc((B+1) * sizeof(int));
//     int kv_len_each = (seq_len / 128);
//     for(int i = 0; i < B; ++i){
//         h_kv_len[i] = kv_len_each;
//         kv_start_offsets[i] = total_kv_len;
//         total_kv_len += h_kv_len[i];
//     }
//     kv_start_offsets[B] = total_kv_len;
//     float *h_score;
//     h_score = (float*)malloc(total_kv_len * sizeof(float));
//
//     int *block_table;
//     block_table = (int*)malloc(total_kv_len * sizeof(int));
//     for(int i = 0; i < total_kv_len; ++i){
//         block_table[i] = i * 5 % N;
//     }
//     int *batch_index;
//     batch_index = (int*)malloc(total_kv_len * sizeof(int));
//
//     for(int i = 0, j = 0; i < total_kv_len; ++i){
//         if(i < kv_start_offsets[j+1] && i >= kv_start_offsets[j]){
//             batch_index[i] = j;
//         }
//         else{
//             ++j;
//             batch_index[i] = j;
//         }
//     }
//
//
//     float *d_Q, *d_K, *d_score;
//     cuda_check(cudaMalloc(&d_Q, sizeof(float) * B * dim));
//     cuda_check(cudaMalloc(&d_K, sizeof(float) * N * dim));
//     cuda_check(cudaMalloc(&d_score, sizeof(float) * total_kv_len));
//     cuda_check(cudaMemcpy(d_Q, h_Q, sizeof(float) * B * dim, cudaMemcpyHostToDevice));
//     cuda_check(cudaMemcpy(d_K, h_K, sizeof(float) * N * dim, cudaMemcpyHostToDevice));
//
//     int *d_block_table, *d_batch_index;
//     cuda_check(cudaMalloc(&d_block_table, sizeof(int) * total_kv_len));
//     cuda_check(cudaMemcpy(d_block_table, block_table, sizeof(int) * total_kv_len, cudaMemcpyHostToDevice));
//     cuda_check(cudaMalloc(&d_batch_index, sizeof(int) * total_kv_len));
//     cuda_check(cudaMemcpy(d_batch_index, batch_index, sizeof(int) * total_kv_len, cudaMemcpyHostToDevice));
//
//
//     dim3 numThreads = {(unsigned int)(32)};
//     dim3 numBlocks = {(unsigned int)total_kv_len};
//
//     for (int i = 0; i < 10; ++i){
//         retrieval_kernel<<<numBlocks, numThreads>>>(d_Q, d_K, d_score, d_block_table, d_batch_index, dim, total_kv_len);
//         // size_t bytes = 2 * dim * sizeof(float) + numThreads.x * sizeof(float);
//         // retrieval_kernel_2<<<numBlocks, numThreads, bytes>>>(d_Q, d_K, d_score, d_block_table, d_batch_index, dim, total_kv_len);
//         size_t bytes = numThreads.x * sizeof(float);
//         retrieval_kernel_3<<<numBlocks, numThreads, bytes>>>(d_Q, d_K, d_score, d_block_table, d_batch_index, dim, total_kv_len);
//     }
//
//     cudaEvent_t start, stop, start_2, stop_2;
//     cudaEventCreate(&start);
//     cudaEventCreate(&stop);
//     cudaEventCreate(&start_2);
//     cudaEventCreate(&stop_2);
//
//
//     cudaEventRecord(start, 0);
//     retrieval_kernel<<<numBlocks, numThreads>>>(d_Q, d_K, d_score, d_block_table, d_batch_index, dim, total_kv_len);
//     cudaEventRecord(stop, 0);
//     cuda_check(cudaPeekAtLastError());
//     cuda_check(cudaEventSynchronize(stop));
//     float milliseconds = 0;
//     cudaEventElapsedTime(&milliseconds, start, stop);
//     printf("Time spent on retrieval_kernel: %f ms\n", milliseconds);
//
//
//     cudaEventRecord(start_2, 0);
//     // size_t bytes = 2 * dim * sizeof(float) + numThreads.x * sizeof(float);
//     // retrieval_kernel_2<<<numBlocks, numThreads, bytes>>>(d_Q, d_K, d_score, d_block_table, d_batch_index, dim, total_kv_len);
//     size_t bytes = numThreads.x * sizeof(float);
//     retrieval_kernel_3<<<numBlocks, numThreads, bytes>>>(d_Q, d_K, d_score, d_block_table, d_batch_index, dim, total_kv_len);
//     cudaEventRecord(stop_2, 0);
//     cuda_check(cudaPeekAtLastError());
//     cuda_check(cudaEventSynchronize(stop_2));
//     float milliseconds_2 = 0;
//     cudaEventElapsedTime(&milliseconds_2, start_2, stop_2);
//     printf("Time spent on retrieval_kernel_3: %f ms\n", milliseconds_2);
//
//
//     float *h_score_gpu;
//     h_score_gpu = (float*)malloc(total_kv_len * sizeof(float));
//     cuda_check(cudaMemcpy(h_score_gpu, d_score, total_kv_len * sizeof(float), cudaMemcpyDeviceToHost));
//
//     for (int i = 0; i < 10; ++i){
//         retrieval_host(h_Q, h_K, h_score, block_table, batch_index, dim, total_kv_len);
//     }
//
//     auto h_start = std::chrono::high_resolution_clock::now();
//     retrieval_host(h_Q, h_K, h_score, block_table, batch_index, dim, total_kv_len);
//     auto h_stop = std::chrono::high_resolution_clock::now();
//     auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(h_stop - h_start);
//     printf("Time spent on retrieval_host: %f ms\n", (float)duration.count() / 1000000.0f);
//
//     float eps = 1e-3;
//     float avg_error = 0.0f;
//     for(int i = 0; i < total_kv_len; ++i){
//         float diff = fabs(h_score[i] - h_score_gpu[i]);
//         avg_error += diff;
//         if(diff > eps){
//             printf("not ok @%d!!! %f vs %f, err %f\n", i, h_score[i], h_score_gpu[i], diff);
//         }
//     }
//     avg_error = avg_error / total_kv_len;
//     printf("avg error: %f\n", avg_error);
//
//     return 0;
// }
