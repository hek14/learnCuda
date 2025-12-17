import numpy as np
import torch
from torch.utils.cpp_extension import load
import pytest
import time

torch.set_grad_enabled(False)
# Load the CUDA kernel as a python module
esa_lib = load(
    name="esa_interface",
    sources=["esa_interface.cu"],
    extra_cflags=["-std=c++17"],
)
esa_retrieval = esa_lib.esa_retrieval
esa_topk = esa_lib.esa_topk
esa_repre = esa_lib.esa_repre

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
{' '*4}number of key blocks for each request: {num_repre_blocks // batch_size}
{' '*4}heads: {num_q_heads}\n''')
    N = num_repre_blocks * 2
    num_k_heads = 8
    query_list = []
    dtype = torch.bfloat16
    for i in range(batch_size):
        query_list.append(torch.rand(num_q_heads, dim, dtype=dtype).cuda())
    repre_cache = torch.randn(N, num_k_heads, dim, dtype = dtype).cuda()
    rng = np.random.default_rng()
    range_n = np.arange(N)
    repre_index = rng.choice(range_n, size=num_repre_blocks, replace=False)
    repre_index = torch.from_numpy(repre_index).to(torch.int32).cuda()
    q_index = torch.randint(0, batch_size, size = [num_repre_blocks], dtype = torch.int32).cuda()
    score = torch.zeros(num_repre_blocks, dtype = dtype).cuda()
    score_sorted = torch.zeros(num_repre_blocks, dtype = dtype).cuda()
    index = torch.cat([torch.arange(0, num_repre_blocks / batch_size, dtype=torch.int32) for _ in range(batch_size)]).cuda()
    index_sorted = torch.arange(0, num_repre_blocks, dtype=torch.int32).cuda()
    batch_offset = torch.arange(0, num_repre_blocks, num_repre_blocks / batch_size, dtype=torch.int32).cuda()
    batch_offset = torch.cat([batch_offset, torch.tensor([num_repre_blocks], dtype=torch.int32).cuda()])
    workspace = torch.zeros(10000, dtype=torch.int32).cuda()

    Input = esa_lib.RetrievalInputTensor()
    Input.num_q_heads = num_q_heads;
    Input.q_ptrs = torch.tensor([q.data_ptr() for q in query_list],
                                dtype=torch.int64, device='cuda')
    Input.repre_cache = repre_cache
    Input.q_index = q_index
    Input.repre_index = repre_index
    Input.batch_offset = batch_offset
    Input.workspace = workspace

    Output = esa_lib.RetrievalOutputTensor()
    Output.score = score
    Output.score_sorted = score_sorted
    Output.index_ranged = index
    Output.index_sorted = index_sorted

    start = time.perf_counter_ns()
    esa_retrieval(Input, Output)
    torch.cuda.synchronize()
    duration = time.perf_counter_ns() - start
    print_green(f"{' '*4}esa_retrieval host API time: {duration/1e6:.3f} ms")

    def naive_retrieval():
        query = torch.stack(query_list)
        query = query[q_index]
        key = torch.repeat_interleave(repre_cache[repre_index], num_q_heads//num_k_heads, dim=1)
        score_gt = (query * key).sum(-1).sum(-1)
        index_gt = torch.cat([ score_gt[s:t].argsort(descending=True) for s,t in zip(batch_offset[:-1], batch_offset[1:]) ])
        return score_gt, index_gt

    start = time.perf_counter_ns()
    score_gt, index_gt = naive_retrieval()
    torch.cuda.synchronize()
    duration = time.perf_counter_ns() - start
    print_red(f"{' '*4}naive_retrieval host API time: {duration/1e6:.3f} ms")

    diff = (score - score_gt).abs()
    print_blue(f"{' '*4}score diff: {diff.mean():.3f}(mean), {diff.max():.3f}(max)")
    # diff_index = (index_sorted - index_gt).abs().to(torch.float32)
    # print_blue(f"{' '*4}index diff: {diff_index.mean():.3f}(mean), {diff_index.max():.3f}(max)")
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
