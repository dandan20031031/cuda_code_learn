# -*- coding: utf-8 -*-
"""
PyTorch Attention 基线：给我们的 CUDA v2.0 做参照
==================================================
和 flashattention_bench.cu 同一组配置（B=4 H=8 D=64），三个基线：

  1. torch naive (FP32) —— 「不用 flash 的人」真实写法：
     softmax(Q@K^T / sqrt(d)) @ V，cuBLAS GEMM + 独立 softmax kernel，
     中间 S/P 落 HBM。这是公平基线（我们 bench 里的 naive golden 算了
     3 遍 QK^T，故意写弱了，不能当 perf 基线）。

  2. SDPA BF16 —— F.scaled_dot_product_attention，flash/cuDNN 后端
     走 Tensor Core。这是「商用天花板」，衡量我们的 v2.0 离工业界多远。

  3. SDPA FP32 —— 看 FP32 输入会 fallback 到哪个后端。

计时口径和 C++ 侧一致：预热 + CUDA event，TFLOPS 按 4·N²·D（causal 减半）。
"""

import torch
import warnings
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend

warnings.filterwarnings("ignore")   # SDPA 后端选择的 UserWarning 刷屏，关掉
torch.manual_seed(0)

B, H, D = 4, 8, 64
NS = [1024, 2048, 4096, 8192]
WARMUP, REPEAT = 3, 10


def bench(fn, *args, **kw):
    """CUDA event 计时，返回 ms"""
    for _ in range(WARMUP):
        fn(*args, **kw)
    torch.cuda.synchronize()
    t0 = torch.cuda.Event(enable_timing=True)
    t1 = torch.cuda.Event(enable_timing=True)
    t0.record()
    for _ in range(REPEAT):
        fn(*args, **kw)
    t1.record()
    torch.cuda.synchronize()
    return t0.elapsed_time(t1) / REPEAT


def tflops(N, causal, ms):
    flops = 4.0 * B * H * N * N * D
    if causal:
        flops *= 0.5
    return flops / (ms * 1e-3) / 1e12


def _causal_mask(n, device):
    """bool 上三角 mask，按 N 缓存（N=8192 时 67MB，只建一次）"""
    key = (n, device)
    if key not in _mask_cache:
        _mask_cache[key] = torch.triu(
            torch.ones(n, n, device=device, dtype=torch.bool), diagonal=1)
    return _mask_cache[key]


def _chunked_materialized(q, k, v, causal=False):
    """手写 non-flash：物化 S/P，中间结果走 HBM —— 公平基线。

    【必须按 head 分块】：一口气跑全部 B*H=32 个头时，N=8192 的
    S+P = 2 × 32 × 8192² × 4B ≈ 17.2 GB，超出 16GB 显存 →
    分配器反复腾挪 8GB 大块 → WDDM 驱动超时(TDR) → 黑屏卡死重启。
    按 head 分块后峰值只有几 GB，算法本身不变（仍是 batched cuBLAS
    GEMM + 独立 softmax kernel，S/P 仍落 HBM，基准意义不变）。
    """
    scale = 1.0 / (D ** 0.5)
    out = torch.empty_like(q)
    BH = q.shape[0] * q.shape[1]
    qf = q.reshape(BH, *q.shape[2:])                   # [BH, N, D]（连续张量，view）
    kf = k.reshape(BH, *k.shape[2:])
    vf = v.reshape(BH, *k.shape[2:])
    of = out.reshape(BH, *q.shape[2:])
    N = q.shape[2]

    # 每个 head 需要物化 S、P 两份 [N,N] fp32；按剩余显存的 50% 定块大小
    free_bytes = torch.cuda.mem_get_info()[0]
    per_head = 2 * N * N * 4
    chunk = max(1, min(BH, int(free_bytes * 0.5) // per_head))

    for st in range(0, BH, chunk):
        ed = min(st + chunk, BH)
        s = torch.matmul(qf[st:ed], kf[st:ed].transpose(-2, -1)) * scale
        if causal:
            s.masked_fill_(_causal_mask(N, s.device), float("-inf"))  # 原地，不多复制一份
        p = torch.softmax(s, dim=-1)
        of[st:ed] = torch.matmul(p, vf[st:ed])
        del s, p                                       # 及早还块，下轮直接复用
    return out


def attn_naive(q, k, v):
    return _chunked_materialized(q, k, v, causal=False)


def attn_sdpa(q, k, v, is_causal=False):
    return F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)


_mask_cache = {}   # causal mask 是常量，缓存避免每次计时都重新分配


def attn_naive_causal(q, k, v):
    """手写 causal：上三角置 -inf 再 softmax（exp(-inf)=0，权重自然为零）"""
    return _chunked_materialized(q, k, v, causal=True)


def main():
    dev = torch.device("cuda")
    print(f"device: {torch.cuda.get_device_name(0)}, "
          f"sm_{torch.cuda.get_device_capability(0)[0]}xx, torch {torch.__version__}")
    print(f"B={B} H={H} D={D}  warmup={WARMUP} repeat={REPEAT}")
    print()

    hdr = f"{'N':>6} | {'naive fp32':>20} | {'SDPA bf16':>20} | {'SDPA fp32':>20}"
    print(hdr)
    print("-" * len(hdr))

    for N in NS:
        # [B,H,N,D] 连续布局，和 C++ bench 一致，view 即可无转置
        q = (torch.rand(B, H, N, D, device=dev) - 0.5)
        k = (torch.rand(B, H, N, D, device=dev) - 0.5)
        v = (torch.rand(B, H, N, D, device=dev) - 0.5)
        q16, k16, v16 = q.bfloat16(), k.bfloat16(), v.bfloat16()

        # 1. naive FP32（公平基线）
        ms_naive = bench(attn_naive, q, k, v)

        # 2. SDPA bf16 —— Windows 轮子没编译 FA2 后端，优先 cuDNN，再试 flash
        #    （都不带 context manager 跑 benchmark，避免异常时全局状态污染）
        backend16 = None
        for name, be in [("cudnn", SDPBackend.CUDNN_ATTENTION),
                         ("flash", SDPBackend.FLASH_ATTENTION)]:
            try:
                with sdpa_kernel([be]):
                    attn_sdpa(q16, k16, v16)           # 试跑一次确认可用
                backend16 = name
                break
            except Exception:
                continue
        if backend16 is None:
            # 两个专属后端都不可用 → 默认后端（可能 math/effective）
            backend16 = "default"
        if backend16 == "cudnn":
            with sdpa_kernel([SDPBackend.CUDNN_ATTENTION]):
                ms_flash16 = bench(attn_sdpa, q16, k16, v16)
        elif backend16 == "flash":
            with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
                ms_flash16 = bench(attn_sdpa, q16, k16, v16)
        else:
            ms_flash16 = bench(attn_sdpa, q16, k16, v16)

        # 3. SDPA fp32 —— 逐后端探测：fp32 不支持 flash/cudnn，但 mem_efficient 支持
        #    （math 后端是分解算子，同算力路径的 attn_naive 只有 ~3 TF；
        #     实测 fp32 能跑 ~17 TF，说明走的是 mem_efficient 的 flash 家族 kernel）
        backend32 = None
        for name, be in [("efficient", SDPBackend.EFFICIENT_ATTENTION),
                         ("math", SDPBackend.MATH)]:
            try:
                with sdpa_kernel([be]):
                    attn_sdpa(q, k, v)                 # 试跑一次确认可用
                backend32 = name
                break
            except Exception:
                continue
        if backend32 == "efficient":
            with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
                ms_flash32 = bench(attn_sdpa, q, k, v)
        else:
            ms_flash32 = bench(attn_sdpa, q, k, v)
            backend32 = backend32 or "default"

        print(f"{N:>6} | {ms_naive:>9.3f} ms {tflops(N, False, ms_naive):>7.2f} TF"
              f" | {ms_flash16:>9.3f} ms {tflops(N, False, ms_flash16):>7.2f} TF [{backend16}]"
              f" | {ms_flash32:>9.3f} ms {tflops(N, False, ms_flash32):>7.2f} TF [{backend32}]")

        # ---- causal（N>=2048 的三个）----
        if N >= 2048:
            ms_naive_c = bench(attn_naive_causal, q, k, v)
            if backend16 == "cudnn":
                with sdpa_kernel([SDPBackend.CUDNN_ATTENTION]):
                    ms_sdpa_c = bench(attn_sdpa, q16, k16, v16, is_causal=True)
            else:
                ms_sdpa_c = bench(attn_sdpa, q16, k16, v16, is_causal=True)
            print(f"       | causal: naive {ms_naive_c:.3f} ms {tflops(N, True, ms_naive_c):.2f} TF"
                  f" | sdpa[{backend16}] {ms_sdpa_c:.3f} ms {tflops(N, True, ms_sdpa_c):.2f} TF")

        del q, k, v, q16, k16, v16
        torch.cuda.empty_cache()

    print()
    print("参照（同一台 5080，C++ bench，v2.0 标量版 Br=128）：")
    print("  N=2048: 1.90 ms | N=4096: 9.0 ms | N=8192: 38.0 ms (14.45 TF)")


if __name__ == "__main__":
    main()
