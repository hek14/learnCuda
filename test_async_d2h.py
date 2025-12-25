import time
import torch
from torch.utils.cpp_extension import load
import torch.cuda.nvtx as nvtx

# Build the CUDA extension from source
async_ext = load(
    name="async_d2h_ext",
    sources=["async_d2h_callback.cu"],
    extra_cflags=["-O3", "-std=c++17"],
    extra_cuda_cflags=["-O3", "-std=c++17"],
    extra_ldflags=["-lnvToolsExt"]
)

def main():
    torch.manual_seed(0)
    device = "cuda"
    N = 500  # number of rows
    D = 1024      # feature dimension
    iters = 5

    handles = []
    host_buffers = []

    # Warmup
    q = torch.randn(N, D, device=device, dtype=torch.float32)
    k = torch.randn(N, D, device=device, dtype=torch.float32)
    host_out = torch.empty(N, dtype=torch.float32, device="cpu", pin_memory=True)
    h = async_ext.launch_async(q, k, host_out)
    handles.append(h)
    host_buffers.append(host_out)


    # Launch loop without synchronizing after D2H
    start = time.time()
    for t in range(iters):
        nvtx.range_push(f"iter_{t:02d}")

        # Create fresh inputs and a dedicated pinned output buffer per iteration to avoid aliasing
        q = torch.randn(N, D, device=device, dtype=torch.float32)
        k = torch.randn(N, D, device=device, dtype=torch.float32)
        host_out = torch.empty(N, dtype=torch.float32, device="cpu", pin_memory=True)

        nvtx.range_push("enqueue_launch_async")
        t0 = time.time()
        h = async_ext.launch_async(q, k, host_out)
        launch_ms = (time.time() - t0) * 1000.0
        nvtx.range_pop()
        print(f"[launch {t:02d}] enqueue time: {launch_ms:.3f} ms, pending={async_ext.pending()}")

        handles.append(h)
        host_buffers.append(host_out)

        # nvtx.range_push("cpu_work")
        # # Do some unrelated CPU work to show we are not blocked by D2H
        # _ = sum(i * i for i in range(10000))
        # nvtx.range_pop()

        nvtx.range_pop()

    enqueue_elapsed = (time.time() - start) * 1000.0
    print(f"Enqueued {len(handles)} operations in {enqueue_elapsed:.2f} ms (no sync).")

    # Non-blocking polling until all callbacks have finished
    nvtx.range_push("polling_for_results")
    remaining = set(handles)
    results = {}
    deadline = time.time() + 30.0
    while remaining and time.time() < deadline:
        done = []
        for h in list(remaining):
            ready, idx, val = async_ext.poll(h)
            if ready:
                results[h] = (idx, val)
                done.append(h)
        for h in done:
            remaining.remove(h)
            async_ext.cleanup(h)
        # Sleep a bit to avoid busy waiting
        time.sleep(0.001)
    if remaining:
        print(f"Timeout waiting for {len(remaining)} handles; cleaning up remaining.")
        for h in list(remaining):
            async_ext.cleanup(h)
        remaining.clear()
    nvtx.range_pop()

    print(f"All callbacks finished. Collected {len(results)} results.")

if __name__ == "__main__":
    main()
