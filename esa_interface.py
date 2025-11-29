import torch
from torch.utils.cpp_extension import load
import time

torch.set_grad_enabled(False)

# Load the CUDA kernel as a python module
lib = load(
    name="retrieval_kernel",
    sources=["retrieval_kernel.cu"],
    extra_cuda_cflags=[
        "-O3",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--use_fast_math",
    ],
    extra_cflags=["-std=c++17"],
)
cuda_retrieval = lib.cuda_retrieval

b = 100
s = 100
dim = 576
N = 1000
query_list = []
for i in range(b):
    query_list.append(torch.rand(dim, dtype=torch.float32).cuda())


repre_cache = torch.randn(N, dim, dtype = torch.float32).cuda()
repre_table = torch.arange(0, s, dtype = torch.int32).cuda()
q_table = torch.arange(0, s, dtype = torch.int32).cuda()
score = torch.zeros(s, dtype = torch.float32).cuda()
cuda_retrieval(query_list, repre_cache, q_table, repre_table, score)
print("score: ", score)


def naive_retrieval():
    query = torch.stack(query_list)
    score_gt = (query[q_table] * repre_cache[repre_table]).sum(-1)
    return score_gt

score_gt = naive_retrieval()

print("score_gt: ", score_gt)

diff = (score - score_gt).abs()
print("diff: ", diff.mean(), diff.max())
