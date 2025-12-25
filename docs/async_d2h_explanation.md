Why the previous version didn’t overlap and why this one does

Problem in the earlier design
- Host callback on the compute stream: We originally used cudaLaunchHostFunc on the same stream that ran the kernel and the async D2H. CUDA stream semantics serialize all operations placed on that stream. A host callback is a stream operation just like a kernel or memcpy. The stream won’t advance to later GPU work until that host function returns. Result: GPU work in iteration t+1 can’t run while the host callback for iteration t is running, so you saw no overlap.
- Heavy work inside the callback: cudaLaunchHostFunc executes on an internal CUDA driver thread. If the callback does heavy CPU work (like a large argmin), it can stall driver progress and block the scheduling/submission of subsequent GPU operations, further reducing overlap.
- Potentially non-async D2H: Asynchronous device-to-host copies truly overlap only when the host destination is pinned memory. You already used pin_memory=True, so that part was fine, but it’s important for real overlap.

What changed to enable overlap
1) Put the host callback on a dedicated callback stream, not the compute stream
   - We record a CUDA event on the compute stream after the kernel and the D2H:
     - cudaEventRecord(ev, compute_stream)
   - We then make a separate, non-blocking callback stream wait on that event:
     - cudaStreamWaitEvent(cb_stream, ev, 0)
   - Finally, we launch the host function on the callback stream:
     - cudaLaunchHostFunc(cb_stream, host_callback, ctx)
   - This preserves dependency (host callback won’t run until the D2H is finished) but decouples the callback from the compute stream. The compute stream is now free to accept and run the next iteration’s kernel immediately. That’s the key reason you now see GPU work for iteration t+1 overlapping with CPU callback work from iteration t.

2) Make the host callback lightweight and offload heavy work to a worker thread
   - The callback now only enqueues a job to our own CPU worker thread and returns quickly. This avoids hogging the CUDA driver’s callback thread.
   - The worker thread performs the heavy argmin on the pinned host buffer, completely independent from CUDA’s internal threads. GPU submission/progress is not impeded.

3) Correct lifetime management so nothing stalls or races
   - Keepalive tensors (q, k, device scores, pinned host buffer) are stored in the Context so they aren’t freed while the GPU or the callback still needs them.
   - We clean up the recorded event in the callback (after the cb_stream waits on it) to avoid leaks.
   - A shutdown() function joins the worker thread so the process exits cleanly.

Why this overlaps on the Nsight Systems timeline
- The compute stream is no longer blocked by the host callback. After the event is recorded, the next iteration’s kernel can be enqueued and executed on the compute stream while:
  - The callback stream waits for the event.
  - The host callback runs, quickly enqueues work to the CPU worker, and returns.
  - The CPU worker performs the heavy argmin in parallel with the GPU’s next kernel/memcpy.
- In the Nsight Systems timeline you should see:
  - GPU kernels and D2H on the compute stream for iteration t+1 starting while:
    - The host callback (NVTX “host_callback_enqueue”) and the CPU worker (NVTX “worker_argmin”) for iteration t are running on CPU threads.
  - This demonstrates true CPU–GPU overlap.

Notes and best practices
- Always use pinned memory for async D2H if you want overlap.
- Avoid heavy work inside cudaLaunchHostFunc; use a lightweight callback that defers to your own worker(s).
- If you must do more operations tied to the completion of a copy, prefer the event + separate stream pattern so you don’t block the main compute stream.
- Clean shutdown prevents stray threads from keeping the process alive.

How to profile it with NVTX and Nsight Systems
- The code is already instrumented with NVTX. You can capture and open a report to verify overlap. From your project root:

```bash
nsys profile -t cuda,nvtx,osrt -o nsys_async python3 test_async_d2h.py
```

- Then open the report file (nsys_async.nsys-rep or .qdrep) with the Nsight Systems UI on a machine with a GUI, or copy the report to your workstation. In the timeline:
  - Look for “enqueue: row_dot kernel” and “enqueue: D2H scores” on the compute stream for iteration t+1 running while:
  - “host_callback_enqueue” and “worker_argmin” are active on CPU threads for iteration t.
  - That overlap confirms the decoupling created by the event + callback stream + CPU worker changes.
