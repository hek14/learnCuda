#include <cstdlib>
#include <cuda_runtime.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#define tile 32

float generateRandomFloat(float min, float max) {
    // Generate a random float between 0.0 and 1.0
    float random_normalized = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    // Scale and shift to the desired range [min, max]
    return min + random_normalized * (max - min);
}

#define cuda_check(call){ \
    cudaError_t err = call; \
    if(err != cudaSuccess){ \
        fprintf(stderr, "cuda error: %s %d %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} \


// [m, k] [k, n]
__global__ void matmul(float* A, float* B, float* C, int m, int n, int k){
    int global_x = blockIdx.x * blockDim.x + threadIdx.x;
    int global_y = blockIdx.y * blockDim.y + threadIdx.y;

    if(global_x < n && global_y < m){
        float tmp = 0.0f;
        for(int i = 0; i < k; ++i){
            tmp += A[global_y * k + i] * B[i * n + global_x];
        }
        C[global_y*n+global_x] = tmp;
    }
}

__global__ void matmul_with_shared_mem(float* A, float *B, float *C, int m, int n, int k){
    __shared__ float sh_a[tile][tile];
    __shared__ float sh_b[tile][tile];
    int local_x = threadIdx.x;
    int local_y = threadIdx.y;
    int global_x = blockIdx.x * tile + local_x;
    int global_y = blockIdx.y * tile + local_y;

    int num_tiles = (k + tile - 1) / tile;
    float res = 0.0f;
    for(int i = 0; i < num_tiles; ++i){
        if (global_y < m && i * tile + local_x < k){
            sh_a[local_y][local_x] = A[global_y * k + i * tile + local_x];
        }
        else{
            sh_a[local_y][local_x] = 0.0f;
        }
        if(i * tile + local_y < k && global_x < n){
            sh_b[local_y][local_x] = B[(i * tile + local_y) * n + global_x];
        }
        else{
            sh_b[local_y][local_x] = 0.0f;
        }
        __syncthreads();
        for(int j = 0; j < tile; ++j){
            res += sh_a[local_y][j] * sh_b[j][local_x];
        }
        __syncthreads();
    }
    if(global_y < m && global_x < n)
        C[global_y * n + global_x] = res;
}



void matmulHost(float* A, float* B, float* C, int m, int n, int k){
    for(int i = 0; i < m; ++i){
        for(int j = 0; j < n; ++j){
            float sum = 0.0f;
            for(int s = 0; s < k; ++s){
                sum += A[i * k + s] * B[s * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

void initMat(float *mat, int rows, int cols){
    for(int i = 0; i < rows; ++i){
        for(int j = 0; j < cols; ++j){
            mat[i * cols + j] = generateRandomFloat(0.0f, 5.0f);
        }
    }
}

bool compare(float* gt, float* pred, int rows, int cols){
    float eps = 1e-3;
    for(int i = 0; i < rows; ++i){
        for(int j = 0; j < cols; ++j){
            if(fabs(gt[i*cols+j] - pred[i*cols+j]) > eps){
                printf("not ok at %d %d, host: %f, device: %f\n", i, j, gt[i*cols+j], pred[i*cols+j]);
                return false;
            }
        }
    }
    return true;
}

int main(){
    srand(static_cast<unsigned int>(time(0)));

    int m = 100, n = 256, k = 700;
    float *h_a, *h_b, *h_c, *h_c_gpu;
    h_a = (float*)malloc(m*k*sizeof(float));
    h_b = (float*)malloc(k*n*sizeof(float));
    h_c = (float*)malloc(m*n*sizeof(float));
    h_c_gpu = (float*)malloc(m*n*sizeof(float));
    initMat(h_a, m, k);
    initMat(h_b, k, n);
    matmulHost(h_a, h_b, h_c, m, n, k);

    float *d_a, *d_b, *d_c;
    cuda_check(cudaMalloc(&d_a, m*k*sizeof(float)));
    cuda_check(cudaMalloc(&d_b, k*n*sizeof(float)));
    cuda_check(cudaMalloc(&d_c, m*n*sizeof(float)));

    cuda_check(cudaMemcpy(d_a, h_a, m*k*sizeof(float), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(d_b, h_b, k*n*sizeof(float), cudaMemcpyHostToDevice));
    dim3 numThreadsPerBlock(tile, tile);
    int num_blocks_x = (n+tile-1)/tile;
    int num_blocks_y = (m+tile-1)/tile;
    dim3 numBlocksPerGrid( num_blocks_x, num_blocks_y );
    matmul_with_shared_mem<<<numBlocksPerGrid, numThreadsPerBlock>>>(d_a, d_b, d_c, m, n, k);
    cuda_check(cudaPeekAtLastError());
    cuda_check(cudaDeviceSynchronize());

    cuda_check(cudaMemcpy(h_c_gpu, d_c, m*n*sizeof(float), cudaMemcpyDeviceToHost));

    if(compare(h_c, h_c_gpu, m, n)){
        printf("ok\n");
    }
    else{
        printf("not ok\n");
    }
    return 0;
}
