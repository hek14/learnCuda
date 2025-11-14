#include <cuda_runtime.h>
#include <stdio.h>
#include <chrono>
#include <random>

#define cuda_check(call){ \
    cudaError_t err = call; \
    if(err != cudaSuccess){ \
        fprintf(stderr, "cuda_error %s %d %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
    } \
} \


__global__ void extract_repre(const float *key_cache, float *repre_cache, const int *block_table, int block_size, int dim, int block_number) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < block_number) {
        int block_id = block_table[idx];
        const float* key_ptr = key_cache + block_id * block_size * dim;
        float* repre_ptr = repre_cache + block_id * dim;
        for (int d = 0; d < dim; ++d) {
            float sum = 0.0f;
            for (int j = 0; j < block_size; ++j) {
                sum += key_ptr[j * dim + d];
            }
            repre_ptr[d] = sum / block_size;
        }
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

void host_extract_repre(const float *key_cache, float *repre_cache, const int *block_table, int block_size, int dim, int block_number){
    for(int idx = 0; idx < block_number; ++idx){
        int block_id = block_table[idx];
        const float* key_ptr = key_cache + block_id * block_size * dim;
        float* repre_ptr = repre_cache + block_id * dim;
        for(int d = 0; d < dim; ++d){
            float sum = 0.0f;
            for(int j = 0; j < block_size; ++j){
                sum += key_ptr[j * dim + d];
            }
            repre_ptr[d] = sum / block_size;
        }
    }
}

int main(){
    int N = 1000;
    int block_size = 128;
    int dim = 64;
    int block_number = N;
    // host allocations
    float *h_key_cache = (float*)malloc(N * block_size * dim * sizeof(float));
    float *h_repre = (float*)malloc(N * dim * sizeof(float));
    float *h_repre_gpu = (float*)malloc(N * dim * sizeof(float));
    int *h_block_table = (int*)malloc(block_number * sizeof(int));

    init_mat(h_key_cache, N * block_size * dim);
    for(int i = 0; i < block_number; ++i){
        h_block_table[i] = i;
    }

    // device allocations
    float *d_key_cache, *d_repre;
    int *d_block_table;
    cuda_check(cudaMalloc(&d_key_cache, N * block_size * dim * sizeof(float)));
    cuda_check(cudaMalloc(&d_repre, N * dim * sizeof(float)));
    cuda_check(cudaMalloc(&d_block_table, block_number * sizeof(int)));
    cuda_check(cudaMemcpy(d_key_cache, h_key_cache, N * block_size * dim * sizeof(float), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(d_block_table, h_block_table, block_number * sizeof(int), cudaMemcpyHostToDevice));

    // warm‐up
    int threads = 256;
    int blocks = (block_number + threads - 1) / threads;
    for(int i = 0; i < 10; ++i){
        extract_repre<<<blocks, threads>>>(d_key_cache, d_repre, d_block_table, block_size, dim, block_number);
    }

    // timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    extract_repre<<<blocks, threads>>>(d_key_cache, d_repre, d_block_table, block_size, dim, block_number);
    cudaEventRecord(stop, 0);
    cuda_check(cudaPeekAtLastError());
    cuda_check(cudaEventSynchronize(stop));
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    printf("Time spent on extract_repre: %f ms\n", ms);

    // copy back and verify
    cuda_check(cudaMemcpy(h_repre_gpu, d_repre, N * dim * sizeof(float), cudaMemcpyDeviceToHost));
    host_extract_repre(h_key_cache, h_repre, h_block_table, block_size, dim, block_number);
    float avg_err = 0.0f;
    for(int i = 0; i < N * dim; ++i){
        avg_err += fabs(h_repre[i] - h_repre_gpu[i]);
    }
    avg_err /= (N * dim);
    printf("avg error: %f\n", avg_err);
    return 0;
}

