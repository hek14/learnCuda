#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define cuda_check(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess){ \
        fprintf(stderr, "cuda error: %s: %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

#define tile_width 32

__global__ void mat_transpose(float* output, float* input, int cols, int rows){
    __shared__ float tmp[tile_width][tile_width+1];

    int global_y = blockIdx.y * tile_width + threadIdx.y;
    int local_y = threadIdx.y;

    int global_x = blockIdx.x *  tile_width + threadIdx.x;
    int local_x = threadIdx.x;

    if (global_y < rows && global_x < cols)
        tmp[local_y][local_x] = input[global_y * cols + global_x];

    __syncthreads();

    global_y = blockIdx.x * tile_width + threadIdx.x;
    global_x = blockIdx.y * tile_width + threadIdx.y;
    if(global_y < cols && global_x < rows)
        output[global_y * rows + global_x] = tmp[local_y][local_x];
}

void transposeHost(float *A, float *B, int rows, int cols){
    for(int i = 0; i < rows; ++i){
        for(int j = 0; j < cols; ++j){
            B[j * rows + i] = A[i * cols + j];
        }
    }
}

void initMat(float *mat, int rows, int cols){
    for(int i = 0; i < rows; ++i){
        for(int j = 0; j < cols; ++j){
            mat[i*cols+j] = (float)(i*cols+j);
        }
    }
}

bool compare(float *gpu_res, float *cpu_res, int size){
    float eps = 1e-5;
    for(int i = 0; i < size; ++i){
        if(fabs(gpu_res[i] - cpu_res[i]) > eps){
            fprintf(stderr, "Not ok at index: %d! host: %f, device: %f\n", i, cpu_res[i], gpu_res[i]);
            return false;
        }
    }
    return true;
}

int main(){
    int rows = 1024;
    int cols = 1024;
    int sz = cols * rows;

    float *h_input, *h_output, *h_output_gpu;
    h_input = (float*)malloc(sz * sizeof(float));
    h_output = (float*)malloc(sz * sizeof(float));
    h_output_gpu = (float*)malloc(sz * sizeof(float));
    initMat(h_input, rows, cols);
    transposeHost(h_input, h_output, rows, cols);


    float *d_input, *d_output;
    cuda_check(cudaMalloc(&d_input, sz * sizeof(float)));
    cuda_check(cudaMalloc(&d_output, sz * sizeof(float)));
    cuda_check(cudaMemcpy(d_input, h_input, sz * sizeof(float), cudaMemcpyHostToDevice));

    dim3 threadsPerBlock(tile_width, tile_width);
    dim3 blocksPerGrid(ceil(cols / tile_width), ceil(rows / tile_width));

    mat_transpose<<<blocksPerGrid, threadsPerBlock>>>(d_output, d_input, cols, rows);

    cuda_check(cudaPeekAtLastError());
    cuda_check(cudaDeviceSynchronize());
    cuda_check(cudaMemcpy(h_output_gpu, d_output, sz * sizeof(float), cudaMemcpyDeviceToHost));

    if(compare(h_output_gpu, h_output, sz)){
        printf("ok\n");
    }
    else{
        printf("error\n");
    }
    // for(int i = 0; i < cols; ++i){
    //     for(int j = 0; j < rows; ++j){
    //         printf("%f ", h_output_gpu[i*rows+j]);
    //     }
    //     puts("");
    // }

    free(h_input);
    free(h_output);
    free(h_output_gpu);

    cuda_check(cudaFree(d_input));
    cuda_check(cudaFree(d_output));

    return 0;
}
