import numpy as np
import os
import pathlib
import sysconfig
import subprocess
import torch
import pytest
import time

def build_shared():
    # Build interface.so with nvcc using PyTorch headers/libs
    try:
        import torch
        from torch.utils.cpp_extension import include_paths, library_paths
    except Exception as e:
        raise SystemExit("PyTorch is required to build with nvcc. Install with: pip install torch") from e

    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH") or "/usr/local/cuda"
    cuda_inc = os.path.join(cuda_home, "include")
    cuda_lib = os.path.join(cuda_home, "lib64")
    if not os.path.isdir(cuda_inc) or not os.path.isdir(cuda_lib):
        raise SystemExit(f"CUDA not found. Set CUDA_HOME or install to {cuda_home}")

    py_inc = sysconfig.get_paths()["include"]

    # Torch include/library paths
    t_inc = include_paths()  # e.g., [.../torch/include, .../torch/include/torch/csrc/api/include]
    t_lib = library_paths()  # e.g., [.../torch/lib]

    # ABI flag must match the one PyTorch was built with
    cxx11_abi = getattr(torch._C, "_GLIBCXX_USE_CXX11_ABI", 1)
    abi_macro = f"-D_GLIBCXX_USE_CXX11_ABI={int(cxx11_abi)}"

    print("==== nvcc_compile")
    cmd = [
        "nvcc",
        "-O3",
        "-std=c++17",
        "-Xcompiler",
        "-fPIC",
        "-shared",
        "esa_kernels.cu",
        "esa_interface.cc",
        # includes
        "-I" + py_inc,
        "-I" + cuda_inc,
        *[f"-I{p}" for p in t_inc],
        # ABI macro
        abi_macro,
        # libs
        "-L" + cuda_lib,
        *[f"-L{p}" for p in t_lib],
        # rpaths so interface.so can find libs at runtime (use -Xlinker for nvcc)
        "-Xlinker", "-rpath", "-Xlinker", cuda_lib,
        *[arg for p in t_lib for arg in ("-Xlinker", "-rpath", "-Xlinker", p)],
        # link against torch and CUDA runtime
        "-lc10",
        "-lc10_cuda",
        "-ltorch_cpu",
        "-ltorch_cuda",
        "-ltorch",
        "-ltorch_python",
        "-lcudart",
        "-o",
        "esa_interface.so",
    ]
    print("Building interface.so with:", " ".join(cmd))
    subprocess.run(cmd, check=True)

def load_module():
    """
    Load the CUDA extension.

    Preference order:
    1. If USE_TORCH_EXTENSION=1 (default) and torch is available, build/load via torch.utils.cpp_extension.load.
    2. Otherwise, build a local interface.so with nvcc and load it from disk.
    """
    use_torch = os.environ.get("USE_TORCH_EXTENSION", "0") == "1"
    if use_torch:
        print("==== torch_compile")
        try:
            import torch
            from torch.utils.cpp_extension import load as torch_load
            # torch ships pybind11 headers and handles the build toolchain
            mod = torch_load(
                name="interface",
                sources=["esa_interface.cc", "esa_kernels.cu"],
                extra_cflags=["-O3", "-std=c++17"],
                extra_cuda_cflags=["-O3"],
                verbose=True,
            )
            return mod
        except Exception as e:
            print(f"[warn] torch extension build failed, falling back to nvcc: {e}")

    so_path = pathlib.Path(__file__).with_name("esa_interface.so")
    if not so_path.exists():
        build_shared()

    # import importlib.machinery
    # import importlib.util
    # # Load the extension from an explicit path without modifying sys.path
    # loader = importlib.machinery.ExtensionFileLoader("interface", str(so_path))
    # spec = importlib.util.spec_from_loader(loader.name, loader)
    # if spec is None:
    #     raise RuntimeError("Failed to create spec for interface.so")
    # module = importlib.util.module_from_spec(spec)
    # loader.exec_module(module)
    import esa_interface as module
    return module


esa_lib = load_module()
esa_retrieval = esa_lib.esa_retrieval
# esa_topk = esa_lib.esa_topk
# esa_repre = esa_lib.esa_repre
# esa_copy = esa_lib.esa_copy

class style():
    RED = '\033[31m'
    GREEN = '\033[32m'
    BLUE = '\033[94m'
    YELLOW = '\033[93m'
    RESET = '\033[0m'

def print_red(msg):
    print(style.RED + msg + style.RESET)

def print_green(msg):
    print(style.GREEN + msg + style.RESET)

def print_blue(msg):
    print(style.BLUE + msg + style.RESET)

def print_yellow(msg):
    print(style.YELLOW + msg + style.RESET)

@pytest.mark.parametrize("batch_size", [1, 10])
@pytest.mark.parametrize("num_repre_blocks", [50, 100])
@pytest.mark.parametrize("num_q_heads", [8, 16, 40])
def test_esa_retrieval(batch_size, num_repre_blocks, num_q_heads):
    dim = 128
    print(f'''TEST esa_retrieval
{' '*4}total number of queries (a.k.a batch_size): {batch_size}
{' '*4}heads: {num_q_heads}\n''')
    total_blocks = num_repre_blocks * batch_size
    N = total_blocks * 2
    num_k_heads = 8
    dtype = torch.float32
    query = torch.randn(batch_size, num_q_heads, dim, dtype=dtype).cuda()
    repre_cache = torch.randn(N, num_k_heads, dim, dtype = dtype).cuda()
    rng = np.random.default_rng()
    range_n = np.arange(N)
    repre_index = rng.choice(range_n, size=total_blocks, replace=False)
    repre_index = torch.from_numpy(repre_index).to(torch.int32).cuda()
    q_index = torch.randint(0, batch_size, size = [total_blocks], dtype = torch.int32).cuda()
    score = torch.zeros(total_blocks, dtype = dtype).cuda()
    score_sorted = torch.zeros(total_blocks, dtype = dtype).cuda()
    index_ranged = torch.cat([torch.arange(0, num_repre_blocks) for _ in
                              range(batch_size)]).to(torch.int32).cuda()
    index_sorted = torch.arange(0, total_blocks, dtype=torch.int32).cuda()
    batch_offset = []
    for i in range(batch_size + 1):
        batch_offset.append(i * num_repre_blocks)
    batch_offset = torch.tensor(batch_offset, dtype=torch.int32).cuda()
    workspace = torch.zeros(10000, dtype=torch.int32).cuda()
    # ptrs_host = torch.tensor([q.data_ptr() for q in query_list],
    #                          dtype=torch.int64, pin_memory=True)
    # ptrs_dev = torch.zeros(batch_size, dtype=torch.int64, device="cuda")
    # size = ptrs_host.numel() * ptrs_host.element_size()
    # esa_copy(ptrs_host, ptrs_dev, size)
    # print("ptrs: ", ptrs_dev, ptrs_host)

    Input = esa_lib.RetrievalInputTensor()
    Input.num_q_heads = num_q_heads;
    Input.batch_size = batch_size;
    Input.q_ptrs = query
    # Input.q_ptrs = ptrs_dev
    Input.repre_cache = repre_cache
    Input.q_index = q_index
    Input.repre_index = repre_index
    Input.batch_offset = batch_offset
    Input.workspace = workspace

    Output = esa_lib.RetrievalOutputTensor()
    Output.score = score
    Output.score_sorted = score_sorted
    Output.index_ranged = index_ranged
    Output.index_sorted = index_sorted

    start = time.perf_counter_ns()
    print_red(f"{' '*4}batch_offset {Input.batch_offset}")
    esa_retrieval(Input, Output)
    torch.cuda.synchronize()
    duration = time.perf_counter_ns() - start
    print_green(f"{' '*4}esa_retrieval host API time: {duration/1e6:.3f} ms")
    print_red(f"{' '*4}batch_offset {batch_offset}")

    def naive_retrieval():
        query_batched = query[q_index].to(torch.float32)
        key = torch.repeat_interleave(repre_cache[repre_index],
                                      num_q_heads//num_k_heads,
                                      dim=1).to(torch.float32)
        score_gt = (query_batched * key).sum(-1).sum(-1).to(dtype)
        index_gt = torch.cat([ score_gt[s:t].argsort(descending=True) for s,t in zip(batch_offset[:-1], batch_offset[1:]) ])
        return score_gt, index_gt

    start = time.perf_counter_ns()
    score_gt, index_gt = naive_retrieval()
    torch.cuda.synchronize()
    duration = time.perf_counter_ns() - start
    print_red(f"{' '*4}naive_retrieval host API time: {duration/1e6:.3f} ms")

    diff = (score - score_gt).abs()
    # print_blue(f"{' '*4}score diff: {diff.mean():.3f}(mean), {diff.max():.3f}(max)")
    print(f"score: {score}")
    print(f"score_gt: {score_gt}")
    print(f"score diff: {diff.mean()}, {diff.max()}")
    diff_index = (index_sorted - index_gt).abs()
    print(f"index: {index_sorted}")
    print(f"index_gt: {index_gt}")
    print_blue(f"{' '*4}index diff: {diff_index}")
    print("")


# @pytest.mark.parametrize("num_repre_blocks", [100, 500, 1000])
# @pytest.mark.parametrize("dim", [576, 1024])
# def test_esa_repre(num_repre_blocks, dim):# extract repre
#     print(f'''TEST esa_repre
# {' '*4}total number of blocks to extract_repre: {num_repre_blocks}
# {' '*4}dim (num_heads * hidden_size): {dim}\n''')
#     dtype = torch.bfloat16
#     N = 2 * num_repre_blocks
#     block_size = 128
#     key_cache = torch.randn(N, block_size, dim, dtype=dtype).cuda()
#     repre_cache = torch.randn(N, 1, dim, dtype=dtype).cuda()
#     repre_cache2 = torch.randn(N, 1, dim, dtype=dtype).cuda()
#
#     rng = np.random.default_rng()
#     range_n = np.arange(N)
#     repre_index = rng.choice(range_n, size=num_repre_blocks, replace=False)
#     repre_index = torch.from_numpy(repre_index).to(torch.int32).cuda()
#
#     start = time.perf_counter_ns()
#     esa_repre(key_cache, repre_cache, repre_index, repre_index)
#     torch.cuda.synchronize()
#     duration = time.perf_counter_ns() - start
#     print_green(f"{' '*4}[esa_repre] host API time: {duration / 1e6:.3f} ms")
#
#     start = time.perf_counter_ns()
#     for blk_id in repre_index:
#         repre_cache2[blk_id] = key_cache[blk_id].mean(0)
#     torch.cuda.synchronize()
#     duration = time.perf_counter_ns() - start
#     print_red(f"{' '*4}[naive_repre] host API time: {duration / 1e6:.3f} ms")
#
#     diff = (repre_cache2[repre_index] - repre_cache[repre_index]).abs()
#     print_blue(f"{' '*4}[esa_repre] repre diff: {diff.mean():.3f}(mean), {diff.max():.3f}(max)")
#     print("")


if __name__ == "__main__":
    test_esa_retrieval(1, 52, 40)
    # a = torch.randn(1000, 1000, dtype=torch.float32, device="cuda")
    # b = torch.randn(1000, 1000, dtype=torch.float32, device="cuda")
    # c = torch.randn(1000, 1000, dtype=torch.float32, device="cuda")
    # host = torch.randn(100, 128, 128, pin_memory=True, device="cpu", dtype=torch.float32)
    # dev = torch.zeros(100, 128, 128, device="cuda", dtype=torch.float32)
    # size = host.numel() * host.element_size()
    # ptr1 = torch.tensor([host.data_ptr() for _ in range(4)], device="cpu",
    #                     dtype=torch.uint64, pin_memory=True)
    # ptr2 = torch.zeros(4, device="cuda", dtype=torch.uint64)
    # size1 = ptr1.numel() * ptr1.element_size()
    # size2 = ptr2.numel() * ptr2.element_size()
    # print("sizes: ", size1, size2)
    # esa_copy(ptr1, ptr2, size1)
    # print("ptr1: ",ptr1)
    # print("ptr2: ",ptr2)

    # for i in range(10):
    #     esa_copy(host, dev, size)
    #     c = torch.matmul(a, b)
    # with torch.cuda.nvtx.range(f"beginGGG"):
    #     for i in range(100):
    #         esa_copy(host, dev, size)
    #         c = torch.matmul(a, b)
    #     torch.cuda.synchronize()
