#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <algorithm>
#include <vector>
#include <random>
#include <iostream>

// kernel to initialize indices [0..N)
__global__ void init_indices(int* idx, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) idx[i] = i;
}

int main() {
    const int N = 1 << 10;  // per‐batch length
    const int K = 10;
    const int B = 4;        // number of batches
    // 1) generate random data
    std::vector<float> h_data(B * N);
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (int i = 0; i < N; ++i) h_data[i] = dist(rng);

    // 2) allocate and copy to device
    float* d_data;
    int*   d_indices;
    float* d_sorted_vals;
    int*   d_sorted_idx;
    int total = B * N;
    cudaMalloc(&d_data,       total * sizeof(float));
    cudaMalloc(&d_indices,    total * sizeof(int));
    cudaMalloc(&d_sorted_vals,total * sizeof(float));
    cudaMalloc(&d_sorted_idx, total * sizeof(int));
    cudaMemcpy(d_data, h_data.data(), total * sizeof(float), cudaMemcpyHostToDevice);

    // 3) init device indices
    const int TPB = 256;
    int blocks = (total + TPB - 1) / TPB;
    init_indices<<<blocks, TPB>>>(d_indices, total);
    cudaDeviceSynchronize();

    // 4) run CUB segmented radix sort (one segment)
    // 4) run CUB segmented radix sort over B batches
    std::vector<int> h_offsets(B + 1);
    for (int i = 0; i <= B; ++i) h_offsets[i] = i * N;
    int* d_offsets;
    cudaMalloc(&d_offsets, (B + 1) * sizeof(int));
    cudaMemcpy(d_offsets, h_offsets.data(), (B + 1) * sizeof(int), cudaMemcpyHostToDevice);
    void*  d_temp = nullptr;
    size_t temp_bytes = 0;
    cub::DeviceSegmentedRadixSort::SortPairsDescending(
        d_temp, temp_bytes,
        d_data,  d_sorted_vals,
        d_indices, d_sorted_idx,
        total, B, d_offsets, d_offsets + 1);
    cudaMalloc(&d_temp, temp_bytes);
    cub::DeviceSegmentedRadixSort::SortPairsDescending(
        d_temp, temp_bytes,
        d_data,  d_sorted_vals,
        d_indices, d_sorted_idx,
        total, B, d_offsets, d_offsets + 1);

    // 5) copy top-K indices back
    std::vector<int> h_topk(B * K);
    cudaMemcpy(h_topk.data(), d_sorted_idx, B * K * sizeof(int), cudaMemcpyDeviceToHost);

    // 6) compute CPU ground-truth per batch
    std::vector<int> gt_all;
    gt_all.reserve(B * K);
    for (int batch = 0; batch < B; ++batch) {
        std::vector<int> gt(N);
        for (int j = 0; j < N; ++j) gt[j] = batch * N + j;
        std::partial_sort(gt.begin(), gt.begin() + K, gt.end(),
                          [&h_data](int a, int b) { return h_data[a] > h_data[b]; });
        for (int j = 0; j < K; ++j) gt_all.push_back(gt[j]);
    }
    // 7) compare
    bool ok = true;
    for (int i = 0; i < B * K; ++i) {
        if (h_topk[i] != gt_all[i]) { ok = false; break; }
    }
    std::cout << (ok ? "PASS\n" : "FAIL\n");
    
    // 8) cleanup
    cudaFree(d_data);
    cudaFree(d_indices);
    cudaFree(d_sorted_vals);
    cudaFree(d_sorted_idx);
    cudaFree(d_offsets);
    cudaFree(d_temp);

    return ok ? 0 : 1;
}
