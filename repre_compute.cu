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

int main(){

}

