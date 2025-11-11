#include <cstdlib>
#include <cuda_runtime.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>


#define cuda_check(call){ \
    cudaError_t err = call; \
    if(err != cudaSuccess){ \
        fprintf(stderr, "cuda error %s %d %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} \

#define tile 32

void __global__ matmul(float *A, float *B, float *C, int rows, int cols, int k){
    int local_x = threadIdx.x;
    int local_y = threadIdx.y;
    int global_x = blockIdx.x * blockDim.x + threadIdx.x;
    int global_y = blockIdx.y * blockDim.y + threadIdx.y;
    int num_tiles = (k + tile - 1) / tile;
    __shared__ float sa[tile][tile];
    __shared__ float sb[tile][tile];
    float tmp = 0.0f;
    for(int i = 0; i < num_tiles; ++i){
        // move data to shared_mem
        int tiled_k = i * tile;
        if (global_y < rows && tiled_k + local_y < k)
            sa[local_y][local_x] = A[global_y * k + tiled_k + local_x];
        else
            sa[local_y][local_x] = 0;
        if (tiled_k + local_y < k && global_x < cols)
            sb[local_y][local_x] = B[(tiled_k + local_y) * cols + global_x];
        else
            sb[local_y][local_x] = 0;

        __syncthreads();

        for(int j = 0; j < tile; ++j){
            tmp += sa[local_y][j] * sb[j][local_x];
        }

        __syncthreads();
        // ensure all threads work at the same tile index
    }
    if (global_y < rows && global_x < cols)
        C[global_y*cols +  global_x] = tmp;
}

void __global__ matmul4_wrong(float *A, float *B, float *C, int rows, int cols, int k){
    int local_x = threadIdx.x;
    int local_y = threadIdx.y;
    int global_x = blockIdx.x * blockDim.x + threadIdx.x;
    int global_y = blockIdx.y * blockDim.y + threadIdx.y;
    int num_tiles = (k + tile - 1) / tile;
    __shared__ float sa[tile][tile];
    __shared__ float sb[tile][tile];
    float tmp[4][4];
    for(int i = 0; i < 4; ++i){
        for(int j = 0; j < 4; ++j){
            tmp[i][j] = 0;
        }
    }


    for(int t = 0; t < num_tiles; ++t){
        // each thread move 4*4 elements
        int tiled_k = t * tile;
        for(int i = 0; i < 4; ++i){
            for(int j = 0; j < 4; ++j){
                if (global_y*4+i < rows && tiled_k+local_x*4+j < cols)
                    sa[local_y*4+i][local_x*4+j] = A[(global_y*4+i)*k + (tiled_k+local_x*4+j)];
                else
                    sa[local_y*4+i][local_x*4+j] = 0;

                if(global_x*4+j < k && global_x*4+j < cols)
                    sb[local_y*4+i][local_x*4+j] = B[(tiled_k+local_y*4+i)*cols + (global_x*4+j)];
                else
                    sb[local_y*4+i][local_x*4+j] = 0;
            }
        }
        __syncthreads();

        for(int i = 0; i < 4; ++i){
            for(int j = 0; j < 4; ++j){
                for(int s = 0; s < tile; ++s){
                    tmp[i][j] += sa[local_y*4+i][s] * sb[s][local_x*4+j];
                }
            }
        }
        __syncthreads();
    }
    for(int i = 0; i < 4; ++i){
        for(int j = 0; j < 4; ++j){
            if(global_y*4+i < rows && global_x*4+j < cols)
                C[(global_y*4+i)*cols + (global_x*4+j)] = tmp[i][j];
        }
    }
}

// __global__ void matmul4(float* A, float* B, float* C, int m, int n, int k) {
//     __shared__ float sh_a[tile][tile];
//     __shared__ float sh_b[tile][tile];
//
//     int local_x = threadIdx.x;
//     int local_y = threadIdx.y;
//     int global_x = blockIdx.x * tile + local_x * 4;
//     int global_y = blockIdx.y * tile + local_y * 4;
//
//     float tmp[4][4] = {0};
//
//     int num_tiles = (k + tile - 1) / tile;
//     for(int t = 0; t < num_tiles; ++t){
//         // Load a tile of A and B for each sub-row/sub-column
//         #pragma unroll
//         for(int dy = 0; dy < 4; ++dy){
//             int y = local_y * 4 + dy;
//             int gy = global_y + dy;
//             for(int dx = 0; dx < 4; ++dx){
//                 int x = local_x * 4 + dx;
//                 int gx = global_x + dx;
//                 // A loads:      [gy * k + t*tile + x]
//                 if(gy < m && t*tile + x < k)
//                     sh_a[y][x] = A[gy * k + t*tile + x];
//                 else
//                     sh_a[y][x] = 0.0f;
//                 // B loads:      [(t*tile + y) * n + gx]
//                 if(t*tile + y < k && gx < n)
//                     sh_b[y][x] = B[(t*tile + y) * n + gx];
//                 else
//                     sh_b[y][x] = 0.0f;
//             }
//         }
//
//         __syncthreads();
//
//         // Compute the 4x4 sub-block
//         for(int l = 0; l < tile; ++l){
//             for(int dy = 0; dy < 4; ++dy){
//                 for(int dx = 0; dx < 4; ++dx){
//                     int y = local_y * 4 + dy;
//                     int x = local_x * 4 + dx;
//                     tmp[dy][dx] += sh_a[y][l] * sh_b[l][x];
//                 }
//             }
//         }
//         __syncthreads();
//     }
//
//     // Write results back
//     for(int dy = 0; dy < 4; ++dy){
//         int gy = global_y + dy;
//         if(gy < m){
//             for(int dx = 0; dx < 4; ++dx){
//                 int gx = global_x + dx;
//                 if(gx < n)
//                     C[gy * n + gx] = tmp[dy][dx];
//             }
//         }
//     }
// }


void __global__ matmul4_practice(float *A, float *B, float *C, int rows,  int cols, int k){
    __shared__ float sa[tile][tile];
    __shared__ float sb[tile][tile];
    int local_x = threadIdx.x;
    int global_x = blockIdx.x * tile + threadIdx.x * 4;
    int local_y = threadIdx.y;
    int global_y = blockIdx.y * tile + threadIdx.y * 4;
    int num_tiles = (k + tile - 1) / tile;
    for(int i = 0; i < num_tiles; ++i){
        int tiled_k = i * tile;
        // move
        for(int dy = 0; dy < 4; ++dy){
            for(int dx = 0; dx < 4; ++dx){
                int sx = local_x * 4 + dx;
                int sy = local_y * 4 + dy;
                if(global_y + dy < rows && k + tiled_k + sx)
                    sa[sy][sx] = A[(global_y + dy) * k + tiled_k + sx];
                sb[sy][sx] = B[(tiled_k + sy) * cols + global_x + dx];
            }
        }
    }
}

void matmul_host(float *A, float *B, float *C, int rows, int cols, int k){
    for(int i = 0; i < rows; ++i){
        for(int j = 0; j < cols; ++j){
            float tmp = 0.0f;
            for(int s = 0; s < k; ++s){
                tmp += A[i * k + s] * B[s * cols + j];
            }
            C[i * cols + j] = tmp;
        }
    }
}

float compare(float *h, float *d, int sz){
    float avg_error = 0.0f;
    for(int i = 0; i < sz; ++i){
        float err = fabs(h[i] - d[i]);
        avg_error += err;
    }
    return avg_error / sz;
}

float generateRandomFloat(float min, float max) {
    // Generate a random float between 0.0 and 1.0
    float random_normalized = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    // Scale and shift to the desired range [min, max]
    return min + random_normalized * (max - min);
}

void set_mat(float *m, int sz){
    for(int i = 0; i < sz; ++i){
        m[i] = generateRandomFloat(-5, 5);
    }
}


int main(){
    srand(static_cast<unsigned int>(time(0)));
    int rows = 1000, cols = 900, k = 512;
    float *A, *B, *C, *C_gpu;
    A = (float*)malloc(rows * k * sizeof(float));
    B = (float*)malloc(k * cols * sizeof(float));
    C = (float*)malloc(rows * cols * sizeof(float));
    C_gpu = (float*)malloc(rows * cols * sizeof(float));

    set_mat(A, rows*k);
    set_mat(B, k*cols);

    matmul_host(A, B, C, rows, cols, k);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, rows*k*sizeof(float));
    cudaMalloc(&d_B, k*cols*sizeof(float));
    cudaMalloc(&d_C, rows*cols*sizeof(float));

    cuda_check(cudaMemcpy(d_A, A, rows*k*sizeof(float), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(d_B, B, k*cols*sizeof(float), cudaMemcpyHostToDevice));

    dim3 blockSize(tile/4, tile/4);
    // tile is for data block_size, each thread handles 4x4 elements.
    dim3 gridSize( (cols+tile-1)/ tile, (rows+tile-1)/tile );
    for (int i = 0; i < 10; ++i)
        matmul4<<<gridSize, blockSize>>>(d_A, d_B, d_C, rows, cols, k);
    cuda_check(cudaDeviceSynchronize());

    cudaEvent_t start, end;
    cuda_check(cudaEventCreate(&start));
    cuda_check(cudaEventCreate(&end));

    cuda_check(cudaEventRecord(start, 0));
    matmul4<<<gridSize, blockSize>>>(d_A, d_B, d_C, rows, cols, k);
    cuda_check(cudaPeekAtLastError());
    cuda_check(cudaEventRecord(end, 0));
    cuda_check(cudaEventSynchronize(end));

    float spent = 0;
    cuda_check(cudaEventElapsedTime(&spent, start, end));

    cuda_check(cudaMemcpy(C_gpu, d_C, rows*cols*sizeof(float), cudaMemcpyDeviceToHost));

    printf("total avg_error %f\n", compare(C, C_gpu, rows*cols));
    printf("spent %f ms\n", spent);

    free(A);
    free(B);
    free(C);
    free(C_gpu);

    cuda_check(cudaFree(d_A));
    cuda_check(cudaFree(d_B));
    cuda_check(cudaFree(d_C));

    cuda_check(cudaEventDestroy(start));
    cuda_check(cudaEventDestroy(end));
    return 0;
}
