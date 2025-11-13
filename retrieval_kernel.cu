#include <cuda_runtime.h>
#include <stdio.h>
#include <algorithm>
#include <chrono>
#include <vector>
#include <random>

#define cuda_check(call){ \
    cudaError_t err = call; \
    if(err != cudaSuccess){ \
        fprintf(stderr, "cuda_error %s %d %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
    } \
} \


__global__ void retrieval_kernel(float *Q, float *K, float *score, int *block_table, int *batch_index, int dim, int B, int S){
    // Q: [batch, dim], the query tensors
    // K: [N, dim], the key tensors
    // score: [S], the result score values
    // block_table: [S], the index for K. It's a flattened tensors which actually compose `batch` segment: [N1, N2, N3] for batch = 3, N1 + N2 + N3 = S
    // batch_index: [S], the mark specifying which batch current index belongs to and which Q current K[index] should be compared with.
    // dim: feature size
    int global_x = blockIdx.x * blockDim.x + threadIdx.x;
    int k_index = block_table[global_x];
    int batch_id = batch_index[global_x];
    float s = 0.0f;
    for(int i = 0; i < dim; ++i){
        s += Q[batch_id * dim + i] * K[k_index * dim + i];
    }
    score[global_x] = s;
}


void retrieval_host(float *Q, float *K, float *score, int *block_table, int *batch_index, int dim, int B, int S){
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

int main(){
    float *h_Q, *h_K;
    int B = 3, dim = 8 * 128;
    int N = 1000;

    h_Q = (float*)malloc(B * dim * sizeof(float));
    h_K = (float*)malloc(N * dim * sizeof(float));

    init_mat(h_Q, B * dim);
    init_mat(h_K, N * dim);

    int total_kv_len = 0;
    int *h_kv_len;
    h_kv_len = (int*)malloc(B * sizeof(int));
    int *kv_start_offsets ;
    kv_start_offsets = (int*)malloc((B+1) * sizeof(int));
    for(int i = 0; i < B; ++i){
        h_kv_len[i] = B + i;
        kv_start_offsets[i] = total_kv_len;
        total_kv_len += h_kv_len[i];
    }
    kv_start_offsets[B] = total_kv_len;
    float *h_score;
    h_score = (float*)malloc(total_kv_len * sizeof(float));

    int *block_table;
    block_table = (int*)malloc(total_kv_len * sizeof(int));
    for(int i = 0; i < total_kv_len; ++i){
        block_table[i] = i;
    }
    int *batch_index;
    batch_index = (int*)malloc(total_kv_len * sizeof(int));
    printf("sanity check\n");
    for(int i = 0; i <= B; ++i){
        printf("offset: %d\n", kv_start_offsets[i]);
    }
    printf("total_kv_len: %d\n", total_kv_len);
    for(int i = 0, j = 0; i < total_kv_len; ++i){
        if(i < kv_start_offsets[j+1] && i >= kv_start_offsets[j]){
            batch_index[i] = j;
        }
        else{
            ++j;
            batch_index[i] = j;
        }
        printf("batch_index[%d] = %d\n", i, batch_index[i]);
    }


    float *d_Q, *d_K;
    cuda_check(cudaMalloc(&d_Q, sizeof(float) * B * dim));
    cuda_check(cudaMalloc(&d_K, sizeof(float) * N * dim));
    cuda_check(cudaMemcpy(d_Q, h_Q, sizeof(float) * B * dim, cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(d_K, h_K, sizeof(float) * N * dim, cudaMemcpyHostToDevice));
    int *d_block_table, *d_batch_index;
    cuda_check(cudaMalloc(&d_block_table, sizeof(int) * total_kv_len));
    cuda_check(cudaMemcpy(d_block_table, block_table, sizeof(int) * total_kv_len, cudaMemcpyHostToDevice));
    cuda_check(cudaMalloc(&d_batch_index, sizeof(int) * total_kv_len));
    cuda_check(cudaMemcpy(d_block_table, block_table, sizeof(int) * total_kv_len, cudaMemcpyHostToDevice));


    dim3 numThreads = {32};
    int num_block = (total_kv_len + 31) / 32;
    dim3 numBlocks = {(unsigned int)num_block};
    retrieval_kernel<<<numBlocks, numThreads>>>(d_Q, d_K, d_score, d_block_table, d_batch_index, dim, B, total_kv_len);

    return 0;
}
