#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <algorithm>
#include <ratio>
#include <vector>
#include <random>
#include <iostream>
#include <chrono>

#define cuda_check(call){ \
    cudaError_t err = call; \
    if(err != cudaSuccess){ \
        fprintf(stderr, "cuda error %s %d %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
    } \
} \

// kernel to initialize indices [0..N)
__global__ void init_indices(int* idx, int total, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) idx[i] = i % N;
}

int main() {
    int N, K, B;
    scanf("%d%d%d", &B, &N, &K);
    // 1) generate random data
    std::vector<float> h_data(B * N);
    std::mt19937 rng(123 * 10 + B);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for(int i = 0; i < B * N; ++i){
        h_data[i] = dist(rng);
    }

    // 2) allocate and copy to device
    float* d_data;
    int*   d_indices;
    float* d_sorted_vals;
    int*   d_sorted_idx;
    int total = B * N;
    cuda_check(cudaMalloc(&d_data,       total * sizeof(float)));
    cuda_check(cudaMalloc(&d_indices,    total * sizeof(int)));
    cuda_check(cudaMalloc(&d_sorted_vals,total * sizeof(float)));
    cuda_check(cudaMalloc(&d_sorted_idx, total * sizeof(int)));
    cuda_check(cudaMemcpy(d_data, h_data.data(), total * sizeof(float), cudaMemcpyHostToDevice));

    cudaEvent_t total_start, total_stop, start, stop, start2, stop2;
    float milliseconds = 0;
    float milliseconds_2 = 0;
    float milliseconds_total = 0;
    cuda_check(cudaEventCreate(&total_start));
    cuda_check(cudaEventCreate(&total_stop));
    cuda_check(cudaEventCreate(&start));
    cuda_check(cudaEventCreate(&stop));
    cuda_check(cudaEventCreate(&start2));
    cuda_check(cudaEventCreate(&stop2));



    cudaEventRecord(total_start);
    // 3) init device indices
    const int TPB = 256;
    int blocks = (total + TPB - 1) / TPB;
    init_indices<<<blocks, TPB>>>(d_indices, total, N);

    // 4) run CUB segmented radix sort (one segment)
    // 4) run CUB segmented radix sort over B batches
    std::vector<int> h_offsets(B + 1);
    for (int i = 0; i <= B; ++i) h_offsets[i] = i * N;
    int* d_offsets;
    cuda_check(cudaMalloc(&d_offsets, (B + 1) * sizeof(int)));
    cuda_check(cudaMemcpy(d_offsets, h_offsets.data(), (B + 1) * sizeof(int), cudaMemcpyHostToDevice));

    cudaEventRecord(start);
    void*  d_temp = nullptr;
    size_t temp_bytes = 0;
    cub::DeviceSegmentedRadixSort::SortPairsDescending(
        d_temp, temp_bytes,
        d_data,  d_sorted_vals,
        d_indices, d_sorted_idx,
        total, B, d_offsets, d_offsets + 1);
    cudaEventRecord(stop);
    cuda_check(cudaEventSynchronize(stop));
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("d_temp bytes: %d, number: %d\n", temp_bytes, temp_bytes / sizeof(float));

    cuda_check(cudaMalloc(&d_temp, temp_bytes));

    cudaEventRecord(start2);
    cub::DeviceSegmentedRadixSort::SortPairsDescending(
        d_temp, temp_bytes,
        d_data,  d_sorted_vals,
        d_indices, d_sorted_idx,
        total, B, d_offsets, d_offsets + 1);
    cudaEventRecord(stop2);
    cuda_check(cudaEventSynchronize(stop2));
    cudaEventElapsedTime(&milliseconds_2, start2, stop2);

    printf("kernel spent: %f + %f = %f ms\n", milliseconds, milliseconds_2, milliseconds + milliseconds_2);

    cudaEventRecord(total_stop);
    cudaEventSynchronize(total_stop);
    cudaEventElapsedTime(&milliseconds_total, total_start, total_stop);

    printf("total radix sort spent: %f ms\n", milliseconds_total);

    // 5) copy top-K indices back
    std::vector<int> h_topk(B * N);
    cuda_check(cudaMemcpy(h_topk.data(), d_sorted_idx, B * N * sizeof(int), cudaMemcpyDeviceToHost));

    // 6) compute CPU ground-truth per batch
    auto h_start = std::chrono::high_resolution_clock::now();
    std::vector<int> gt_all;
    for (int batch = 0; batch < B; ++batch) {
        std::vector<int> gt(N);
        for (int j = 0; j < N; ++j) gt[j] = j;
        std::partial_sort(gt.begin(), gt.begin() + K, gt.end(),
                          [&](int a, int b) { return h_data[ batch * N + a] > h_data[ batch *N + b]; });
        for (int j = 0; j < K; ++j) gt_all.push_back(gt[j]);
    }
    auto h_stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(h_stop - h_start);
    printf("host spent: %f\n", (float)duration.count() / 1e6);
    // 7) compare
    for (int i = 0; i < B; ++i) {
        int cnt = 0;
        for(int j = 0; j < K; ++j){
            if (gt_all[i * K + j] != h_topk[i * N + j]){
                cnt += 1;
            }
        }
        printf("batch compare %d non-equal: %d\n", i, cnt);

    }

    // 8) cleanup
    cuda_check(cudaFree(d_data));
    cuda_check(cudaFree(d_indices));
    cuda_check(cudaFree(d_sorted_vals));
    cuda_check(cudaFree(d_sorted_idx));
    cuda_check(cudaFree(d_offsets));
    cuda_check(cudaFree(d_temp));

    return 0;
}
