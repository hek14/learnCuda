#include <cuda_runtime.h>
#include <stdio.h>

// Heavy compute kernel
__global__ void heavy_compute(float *out, size_t N, int repeat) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float sum = 0.0f;
        for (int r = 0; r < repeat; ++r)
            sum += sinf(idx) * cosf(r);
        out[idx] = sum;
    }
}

void run_experiment(int num_copies, size_t chunk_size, int kernel_repeat) {
    float **d_bufs = new float*[num_copies];
    float **h_bufs = new float*[num_copies];
    cudaStream_t copy_stream, compute_stream;
    cudaStreamCreate(&copy_stream);
    cudaStreamCreate(&compute_stream);

    // Allocate device/host buffers
    for (int i = 0; i < num_copies; ++i) {
        cudaMalloc(&d_bufs[i], chunk_size * sizeof(float));
        cudaMallocHost(&h_bufs[i], chunk_size * sizeof(float));
    }

    // Prepare compute buffer
    float *d_compute;
    cudaMalloc(&d_compute, chunk_size * sizeof(float));

    // Init compute buffer
    cudaMemsetAsync(d_compute, 0, chunk_size * sizeof(float), compute_stream);
    cudaDeviceSynchronize();

    // Start timing
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    // Enqueue many DtoH copies (device → host) in copy_stream
    for (int i = 0; i < num_copies; ++i)
        cudaMemcpyAsync(h_bufs[i], d_bufs[i], chunk_size * sizeof(float), cudaMemcpyDeviceToHost, copy_stream);

    // Record twice: before and after kernel launch in compute_stream
    cudaEventRecord(start, compute_stream);
    heavy_compute<<<chunk_size/256, 256, 0, compute_stream>>>(d_compute, chunk_size, kernel_repeat);
    cudaEventRecord(end, compute_stream);

    // Wait for kernel to finish
    cudaEventSynchronize(end);

    // Get timing (ms)
    float kernel_ms = 0.0f;
    cudaEventElapsedTime(&kernel_ms, start, end);

    printf("copies: %d,\tkernel time: %f ms\n", num_copies, kernel_ms);

    // Cleanup
    for (int i = 0; i < num_copies; ++i) {
        cudaFree(d_bufs[i]);
        cudaFreeHost(h_bufs[i]);
    }
    cudaFree(d_compute);
    cudaStreamDestroy(copy_stream);
    cudaStreamDestroy(compute_stream);
    cudaEventDestroy(start);
    cudaEventDestroy(end);
    delete[] d_bufs;
    delete[] h_bufs;
}


void run_experiment_2(int num_copies, size_t chunk_size, int kernel_repeat) {
    float **d_bufs = new float*[num_copies];
    float **h_bufs = new float*[num_copies];
    cudaStream_t copy_stream, compute_stream;
    cudaStreamCreate(&copy_stream);
    cudaStreamCreate(&compute_stream);

    // Allocate device/host buffers
    for (int i = 0; i < num_copies; ++i) {
        cudaMalloc(&d_bufs[i], chunk_size * sizeof(float));
        cudaMallocHost(&h_bufs[i], chunk_size * sizeof(float));
    }

    // Prepare compute buffer
    float *d_compute;
    cudaMalloc(&d_compute, chunk_size * sizeof(float));

    // Init compute buffer
    cudaMemsetAsync(d_compute, 0, chunk_size * sizeof(float), compute_stream);
    cudaDeviceSynchronize();

    // Start timing
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start, 0);

    // Enqueue many DtoH copies (device → host) in copy_stream
    for (int i = 0; i < num_copies; ++i)
        cudaMemcpyAsync(h_bufs[i], d_bufs[i], chunk_size * sizeof(float), cudaMemcpyDeviceToHost, copy_stream);

    heavy_compute<<<chunk_size/256, 256, 0, compute_stream>>>(d_compute, chunk_size, kernel_repeat);

    // cudaStreamSynchronize(copy_stream);
    cudaStreamSynchronize(compute_stream);
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);


    // Get timing (ms)
    float total_ms = 0.0f;
    cudaEventElapsedTime(&total_ms, start, end);

    printf("copies: %d,\tkernel time: %f ms\n", num_copies, total_ms);

    // Cleanup
    for (int i = 0; i < num_copies; ++i) {
        cudaFree(d_bufs[i]);
        cudaFreeHost(h_bufs[i]);
    }
    cudaFree(d_compute);
    cudaStreamDestroy(copy_stream);
    cudaStreamDestroy(compute_stream);
    cudaEventDestroy(start);
    cudaEventDestroy(end);
    delete[] d_bufs;
    delete[] h_bufs;
}


int main() {
    size_t chunk_size = 1 << 12;    // 4096 floats (~16 KB)
    int kernel_repeat = 500000;     // Controls kernel workload

    // Test for different # of copies, observe timing
    int copy_counts[] = {0, 0, 1, 100, 1000, 2000, 5000, 10000};
    printf("==========\ntime spent including kernel launch\n");
    for (int i = 0; i < sizeof(copy_counts)/sizeof(copy_counts[0]); ++i)
        run_experiment_2(copy_counts[i], chunk_size, kernel_repeat);

    printf("==========\ntime spent excluding kernel launch\n");
    for (int i = 0; i < sizeof(copy_counts)/sizeof(copy_counts[0]); ++i)
        run_experiment(copy_counts[i], chunk_size, kernel_repeat);

    return 0;
}

