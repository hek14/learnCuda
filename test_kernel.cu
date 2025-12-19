#include <cuda.h>
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>
static float** cached_d_ptrs = nullptr;
static int    cached_batch  = 0;
static std::vector<float*> cached_host_ptrs = {};

__global__ void add_kernel(float** in, float* out, int n, int batch)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x; 
    if(tid < n){
        float sum = 0.0f;
        for(int i = 0; i < batch; ++i){
            sum += in[i][tid];
        }
        out[tid] = sum;
    }
}


void launch(std::vector<torch::Tensor> in_tensors, torch::Tensor out)
{
    int n = in_tensors[0].numel();
    int batch = in_tensors.size();
    std::vector<float*> ptrs = {};
    for (int i = 0; i < batch; ++i){
        ptrs.push_back(reinterpret_cast<float*>(in_tensors[i].data_ptr()));
    }
    dim3 numThreads = {1024};
    dim3 numBlocks = {(size_t)((n + 1024 - 1) / 1024)};
    printf("%d %d\n", n, batch);
    // reuse or initialize cached device pointer array
    if (cached_batch != batch || ptrs != cached_host_ptrs) {
        if (cached_d_ptrs) cudaFree(cached_d_ptrs);
        cudaMalloc((void**)&cached_d_ptrs, batch * sizeof(float*));
        cudaMemcpy(cached_d_ptrs, ptrs.data(), batch * sizeof(float*), cudaMemcpyHostToDevice);
        cached_batch = batch;
        cached_host_ptrs = ptrs;
    }
    add_kernel<<<numBlocks, numThreads>>>(cached_d_ptrs, out.data_ptr<float>(), n, batch);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch", &launch, "add two CUDA tensors");
}
