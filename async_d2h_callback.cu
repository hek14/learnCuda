#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <nvToolsExt.h>

#include <unordered_map>
#include <mutex>
#include <memory>
#include <atomic>
#include <vector>
#include <limits>
#include <thread>
#include <condition_variable>
#include <deque>

namespace {

__global__ void row_dot_f32(const float* __restrict__ q,
                            const float* __restrict__ k,
                            float* __restrict__ out,
                            int N, int D) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float sum = 0.0f;
        int base = i * D;
        for (int j = 0; j < D; ++j) {
            sum += q[base + j] * k[base + j];
        }
        out[i] = sum;
    }
}

struct Context {
    std::atomic<int> ready;
    int result_idx;
    float result_val;
    int N;

    // Keep memory alive until callback completes
    float* host_ptr;
    torch::Tensor host_tensor_keepalive;   // pinned CPU buffer (destination of D2H)
    torch::Tensor device_out_keepalive;    // device buffer for scores
    torch::Tensor q_keepalive;             // keep input alive
    torch::Tensor k_keepalive;             // keep input alive
    cudaEvent_t ready_event;               // event signaling D2H completion

    Context() : ready(0), result_idx(-1), result_val(std::numeric_limits<float>::infinity()), N(0), host_ptr(nullptr), ready_event(nullptr) {}
};

std::mutex g_mutex;
std::unordered_map<int, std::unique_ptr<Context>> g_contexts;
int g_next_handle = 1;

// Lightweight CPU worker to handle heavy post-D2H work outside CUDA driver threads.
std::mutex w_mutex;
std::condition_variable w_cv;
std::deque<Context*> w_queue;
std::atomic<bool> w_started{false};
std::atomic<bool> w_running{false};
std::thread w_thread;

// Dedicated callback stream to avoid blocking the compute stream with host callbacks.
static cudaStream_t g_cb_stream = nullptr;

inline void ensure_callback_stream() {
    if (g_cb_stream == nullptr) {
        cudaStreamCreateWithFlags(&g_cb_stream, cudaStreamNonBlocking);
    }
}

void worker_loop() {
    // Optional marker that the worker thread started.
    nvtxRangePushA("worker_thread_start");
    nvtxRangePop();

    while (w_running.load(std::memory_order_acquire)) {
        Context* job = nullptr;
        {
            std::unique_lock<std::mutex> lk(w_mutex);
            w_cv.wait(lk, [] {
                return !w_queue.empty() || !w_running.load(std::memory_order_acquire);
            });
            if (!w_running.load(std::memory_order_acquire)) break;
            job = w_queue.front();
            w_queue.pop_front();
        }

        if (job) {
            nvtxRangePushA("worker_argmin");
            float min_val = std::numeric_limits<float>::infinity();
            int min_idx = -1;
            float* data = job->host_ptr;
            int N = job->N;
            for (int i = 0; i < N; ++i) {
                float v = data[i];
                if (v < min_val) {
                    min_val = v;
                    min_idx = i;
                }
            }
            job->result_idx = min_idx;
            job->result_val = min_val;
            job->ready.store(1, std::memory_order_release);
            nvtxRangePop();
        }
    }
}

void ensure_worker_started() {
    bool expected = false;
    if (w_started.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
        w_running.store(true, std::memory_order_release);
        w_thread = std::thread(worker_loop);
        // Do NOT detach: we'll join it explicitly on shutdown to avoid hangs at process exit.
    }
}

void enqueue_job(Context* ctx) {
    {
        std::lock_guard<std::mutex> lk(w_mutex);
        w_queue.push_back(ctx);
    }
    w_cv.notify_one();
}

void CUDART_CB host_callback(void* userData) {
    nvtxRangePushA("host_callback_enqueue");
    Context* ctx = reinterpret_cast<Context*>(userData);
    // Destroy the event now that the callback stream has waited on it.
    if (ctx->ready_event) {
        cudaEventDestroy(ctx->ready_event);
        ctx->ready_event = nullptr;
    }
    // Defer heavy work to a standalone CPU worker thread to avoid blocking CUDA driver threads.
    enqueue_job(ctx);
    nvtxRangePop();
}

} // anonymous namespace

// Launch compute(q,k)->score_d, then async D2H to pinned host_out, then host callback to compute argmin.
// Returns a handle to poll later.
int launch_async(torch::Tensor q, torch::Tensor k, torch::Tensor host_out) {
    TORCH_CHECK(q.is_cuda(), "q must be a CUDA tensor");
    TORCH_CHECK(k.is_cuda(), "k must be a CUDA tensor");
    TORCH_CHECK(q.dim() == 2 && k.dim() == 2, "q and k must be 2D [N,D]");
    TORCH_CHECK(q.sizes() == k.sizes(), "q and k sizes must match");
    TORCH_CHECK(q.scalar_type() == at::kFloat && k.scalar_type() == at::kFloat, "q and k must be float32");
    TORCH_CHECK(!host_out.is_cuda(), "host_out must be a CPU tensor");
    TORCH_CHECK(host_out.is_pinned(), "host_out must be allocated with pin_memory=True");
    TORCH_CHECK(host_out.scalar_type() == at::kFloat, "host_out must be float32");
    TORCH_CHECK(host_out.dim() == 1, "host_out must be 1D of length N");
    TORCH_CHECK(host_out.size(0) == q.size(0), "host_out size must be N");

    int64_t N = q.size(0);
    int64_t D = q.size(1);

    auto q_contig = q.contiguous();
    auto k_contig = k.contiguous();

    auto opts = q.options().device(at::kCUDA);
    auto score_d = torch::empty({N}, opts);

    // Create context and keep tensors alive
    auto ctx = std::make_unique<Context>();
    ctx->N = static_cast<int>(N);
    ctx->host_tensor_keepalive = host_out;  // pin lifetime
    ctx->device_out_keepalive = score_d;    // pin lifetime
    ctx->host_ptr = host_out.data_ptr<float>();
    ctx->q_keepalive = q_contig;            // keep inputs alive until stream work finishes
    ctx->k_keepalive = k_contig;

    int handle;
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        handle = g_next_handle++;
        g_contexts.emplace(handle, std::move(ctx));
    }

    // Use current PyTorch stream
    auto s = at::cuda::getCurrentCUDAStream();
    cudaStream_t stream = s.stream();
    ensure_worker_started();

    // Launch kernel
    nvtxRangePushA("enqueue: row_dot kernel");
    int threads = 256;
    int blocks = (static_cast<int>(N) + threads - 1) / threads;
    row_dot_f32<<<blocks, threads, 0, stream>>>(
        q_contig.data_ptr<float>(),
        k_contig.data_ptr<float>(),
        score_d.data_ptr<float>(),
        static_cast<int>(N),
        static_cast<int>(D)
    );
    cudaError_t kernStatus = cudaPeekAtLastError();
    TORCH_CHECK(kernStatus == cudaSuccess, "row_dot_f32 launch failed: ", cudaGetErrorString(kernStatus));
    nvtxRangePop();

    // Enqueue async D2H
    nvtxRangePushA("enqueue: D2H scores");
    auto& ctx_ref = g_contexts.find(handle)->second;
    cudaError_t memStatus = cudaMemcpyAsync(
        ctx_ref->host_ptr,
        ctx_ref->device_out_keepalive.data_ptr<float>(),
        sizeof(float) * N,
        cudaMemcpyDeviceToHost,
        stream
    );
    TORCH_CHECK(memStatus == cudaSuccess, "cudaMemcpyAsync failed: ", cudaGetErrorString(memStatus));
    nvtxRangePop();

    // Record event on compute stream and enqueue host callback on dedicated callback stream
    nvtxRangePushA("enqueue: record_event_for_host_callback");
    cudaEvent_t ev;
    cudaError_t evCreateStatus = cudaEventCreateWithFlags(&ev, cudaEventDisableTiming);
    TORCH_CHECK(evCreateStatus == cudaSuccess, "cudaEventCreateWithFlags failed: ", cudaGetErrorString(evCreateStatus));
    cudaError_t evRecordStatus = cudaEventRecord(ev, stream);
    TORCH_CHECK(evRecordStatus == cudaSuccess, "cudaEventRecord failed: ", cudaGetErrorString(evRecordStatus));
    ctx_ref->ready_event = ev;
    nvtxRangePop();

    nvtxRangePushA("enqueue: host_callback(cb_stream)");
    ensure_callback_stream();
    cudaError_t waitStatus = cudaStreamWaitEvent(g_cb_stream, ev, 0);
    TORCH_CHECK(waitStatus == cudaSuccess, "cudaStreamWaitEvent failed: ", cudaGetErrorString(waitStatus));
    cudaError_t cbStatus = cudaLaunchHostFunc(g_cb_stream, host_callback, ctx_ref.get());
    TORCH_CHECK(cbStatus == cudaSuccess, "cudaLaunchHostFunc failed: ", cudaGetErrorString(cbStatus));
    nvtxRangePop();

    // Return immediately without synchronization
    return handle;
}

// Non-blocking poll. Returns (ready: bool, min_idx: int, min_val: float).
pybind11::tuple poll(int handle) {
    std::unique_ptr<Context>* pctx = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        auto it = g_contexts.find(handle);
        TORCH_CHECK(it != g_contexts.end(), "invalid handle");
        pctx = &it->second;
    }
    Context* ctx = pctx->get();
    bool ready = ctx->ready.load(std::memory_order_acquire) != 0;
    if (!ready) {
        return pybind11::make_tuple(false, -1, 0.0f);
    }
    return pybind11::make_tuple(true, ctx->result_idx, ctx->result_val);
}

// Optionally release resources for a completed handle.
bool cleanup(int handle) {
    std::lock_guard<std::mutex> lock(g_mutex);
    auto it = g_contexts.find(handle);
    if (it == g_contexts.end()) return false;
    g_contexts.erase(it);
    return true;
}

int pending() {
    std::lock_guard<std::mutex> lock(g_mutex);
    return static_cast<int>(g_contexts.size());
}

void shutdown_worker() {
    // Idempotent shutdown of the CPU worker thread.
    if (w_started.load(std::memory_order_acquire)) {
        w_running.store(false, std::memory_order_release);
        w_cv.notify_all();
        if (w_thread.joinable()) {
            w_thread.join();
        }
        w_started.store(false, std::memory_order_release);
    }
    if (g_cb_stream != nullptr) {
        cudaStreamDestroy(g_cb_stream);
        g_cb_stream = nullptr;
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Async D2H with CUDA host callback demo";
    m.def("launch_async", &launch_async, "Launch async compute + D2H + host callback (returns handle)");
    m.def("poll", &poll, "Poll for result (ready, min_idx, min_val)");
    m.def("cleanup", &cleanup, "Cleanup a handle");
    m.def("pending", &pending, "Number of pending contexts");
    m.def("shutdown", &shutdown_worker, "Stop worker thread and join");
}
