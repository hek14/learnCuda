import time
import torch
from torch.utils.cpp_extension import load

# Build the CUDA extension from source
async_ext = load(
    name="async_d2h_ext",
    sources=["async_d2h_callback.cu"],
    extra_cflags=["-O3", "-std=c++17"],
    extra_cuda_cflags=["-O3", "-std=c++17"]
)

def main():
    torch.manual_seed(0)
    device = "cuda"
    N = 1 << 18  # number of rows
    D = 128      # feature dimension
    iters = 20

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
        q = torch.randn(N, D, device=device, dtype=torch.float32)
        k = torch.randn(N, D, device=device, dtype=torch.float32)
        host_out = torch.empty(N, dtype=torch.float32, device="cpu", pin_memory=True)

        t0 = time.time()
        h = async_ext.launch_async(q, k, host_out)
        launch_ms = (time.time() - t0) * 1000.0
        print(f"[launch {t:02d}] enqueue time: {launch_ms:.3f} ms, pending={async_ext.pending()}")

        handles.append(h)
        host_buffers.append(host_out)

        # Do some unrelated CPU work to show we are not blocked by D2H
        _ = sum(i * i for i in range(10000))

    enqueue_elapsed = (time.time() - start) * 1000.0
    print(f"Enqueued {len(handles)} operations in {enqueue_elapsed:.2f} ms (no sync).")

    # Non-blocking polling until all callbacks have finished
    remaining = set(handles)
    results = {}
    while remaining:
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

    print(f"All callbacks finished. Collected {len(results)} results.")
    # Print a few results
    shown = 0
    for h, (idx, val) in results.items():
        print(f"handle={h} min_idx={idx} min_val={val:.6f}")
        shown += 1
        if shown >= 5:
            break

if __name__ == "__main__":
    main()
