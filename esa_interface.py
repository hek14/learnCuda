import torch
import math
from torch.utils.cpp_extension import load
import time

torch.set_grad_enabled(False)

# Load the CUDA kernel as a python module
lib = load(
    name="esa_utils",
    sources=["esa_utils.cu"],
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
esa_retrieval = lib.esa_retrieval
esa_topk = lib.esa_topk
esa_repre = lib.esa_repre

b = 1
s = 10
dim = 576
N = 100
query_list = []
dtype = torch.float16
for i in range(b):
    query_list.append(torch.rand(dim, dtype=dtype).cuda())


repre_cache = torch.randn(N, dim, dtype = dtype).cuda()
repre_table = torch.arange(0, s, dtype = torch.int32).cuda()
q_table = torch.arange(0, s, dtype = torch.int32).cuda()
q_table = q_table % b
score = torch.zeros(s, dtype = dtype).cuda()
score_sorted = torch.zeros(s, dtype = dtype).cuda()
index = torch.arange(0, s, dtype=torch.int32).cuda()
index_sorted = torch.arange(0, s, dtype=torch.int32).cuda()
offsets = torch.arange(0, s, math.ceil(s / b), dtype=torch.int32).cuda()
offsets = torch.cat([offsets, torch.tensor([s], dtype=torch.int32).cuda()])
workspace = torch.zeros(10000, dtype=torch.int32).cuda()

start = time.time()
esa_retrieval(query_list, repre_cache, q_table, repre_table, score,
              score_sorted, index, index_sorted, offsets, workspace)
print("launch spent: ", time.time() - start)
torch.cuda.synchronize()
elapsed_cuda = time.time() - start
print(f"esa_retrieval time: {elapsed_cuda:.6f} s")


def naive_retrieval():
    query = torch.stack(query_list)
    score_gt = (query[q_table] * repre_cache[repre_table]).sum(-1)
    return score_gt

start = time.time()
score_gt = naive_retrieval()
torch.cuda.synchronize()
elapsed_naive = time.time() - start
print(f"naive_retrieval time: {elapsed_naive:.6f} s")
print("score_gt: ", score_gt)
print("score: ", score)
print("score_sorted: ", score_sorted)
print("index_sorted: ", index_sorted)

diff = (score - score_gt).abs()
print("diff: ", diff.mean(), diff.max())

total_seq_len = 10000
batch_size = 4
topk = 10
num_layers = 61
warmup_iters = 10

score = torch.randn(total_seq_len).cuda()
index = torch.arange(0, total_seq_len, dtype=torch.int32).cuda()
offsets = torch.arange(0, total_seq_len, math.ceil(total_seq_len / batch_size), dtype=torch.int32).cuda()
offsets = torch.cat([offsets, torch.tensor([total_seq_len], dtype=torch.int32).cuda()])


print(f'''info:
==========
Select {topk} from {total_seq_len//batch_size}
batch_size:{batch_size}, num_layers: {num_layers}
==========''')
print("offsets: ", offsets, offsets.shape)

batch_size = offsets.shape[0] - 1
score_out = torch.zeros(total_seq_len).cuda()
index_out = torch.zeros(total_seq_len, dtype=torch.int32).cuda()


cost_time = []
workspace = torch.zeros(10000, dtype=torch.int32).cuda()
for iter in range(warmup_iters + num_layers):
    begin = time.time()
    # reset index tensor for radixSort
    # for i in range(total_seq_len):
    #     index[i] = i
    # for i in range(batch_size + 1):
    #     offsets[i] = i * math.ceil(total_seq_len / batch_size)
    esa_topk(score, index, offsets, score_out, index_out, workspace)

    torch.cuda.synchronize()
    duration = time.time() - begin
    if iter >= warmup_iters:
        cost_time.append(duration)

print(f"esa topk: each_layer: {sum(cost_time) / len(cost_time)}, all_layers: {sum(cost_time)}")


cost_time_2 = []
for iter in range(warmup_iters + num_layers):
    begin = time.time()
    gt_index = []
    for start, stop in zip(offsets[:-1], offsets[1:]):
        sorted, indices = torch.topk(score[start:stop], dim=0, k=topk)
        indices += start
        gt_index.append(indices)
    gt_index = torch.stack(gt_index)
    # score = score.view(batch_size, -1)
    # _, gt_index = torch.topk(score, dim=1, k=topk)
    # gt_index += torch.arange(0, batch_size)[:, None].cuda() * math.ceil(total_seq_len / batch_size)
    gt_index = gt_index.view(-1)
    torch.cuda.synchronize()
    duration = time.time() - begin
    if iter >= warmup_iters:
        cost_time_2.append(duration)

print(f"torch topk: each_layer: {sum(cost_time_2) / len(cost_time_2)}, all_layers: {sum(cost_time_2)}")

index_out = index_out.view(batch_size, -1)
gt_index = gt_index.view(batch_size, -1)
print(index_out.shape, gt_index.shape)
diff = (index_out[:, :topk] - gt_index).abs()
print("diff: ", diff.max())


# extract repre
dtype = torch.bfloat16
key_cache = torch.randn(100, 128, 8, 128, dtype=dtype).cuda()
repre_cache = torch.randn(100, 1, 8, 128, dtype=dtype).cuda()
repre_cache2 = torch.randn(100, 1, 8, 128, dtype=dtype).cuda()
block_table = torch.arange(0, 20, 2, dtype=torch.int32).cuda()
begin = time.time()
esa_repre(key_cache.view(100, 128, -1), repre_cache.view(100, 1, -1), block_table, block_table)
torch.cuda.synchronize()
print("esa_repre spent: ", time.time() - begin)

begin = time.time()
for blk_id in block_table:
    repre_cache2[blk_id] = key_cache[blk_id].mean(0)
torch.cuda.synchronize()
print("normal mean spent: ", time.time() - begin)

diff = (repre_cache2[block_table] - repre_cache[block_table]).abs()
print("repre diff: ", diff.max())
