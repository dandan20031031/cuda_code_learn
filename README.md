# CUDA 编程学习

个人 CUDA 编程学习代码，从零开始掌握 GPU 并行计算。

## SGEMM 优化演进线（`gemm/`）

从 Naive 到逐步优化的 SGEMM（单精度矩阵乘法）实现，每个版本在前一版本基础上改进。

### 性能一览（M=N=K=4096，RTX 5080）

| # | 文件 | 版本 | 关键优化 | 参数 | 耗时 |
|---|------|------|----------|------|------|
| 1 | `ex1.cu` | Hello World | grid/block/thread 基本模型 | — | — |
| 2 | `ex2.cu` | SAXPY | 向量加法 + 错误检查 + 计时 | — | — |
| 3 | `ex3_sgemm_naive.cu` | Naive SGEMM | 最基础实现，HBM 直读 | block=16×16 | 38.3 ms |
| 4 | `ex4_sgemm_tiled.cu` | Shared Memory Tiling | 搬进 shared memory 复用 | block=32×32, BK=32 | 28.6 ms |
| 5 | `ex5_coarsen.cu` | Thread Coarsening | 每线程多输出 + K-tile 扩大 | TILE=16, STRIDE=2, BK=32, 8KB shared | 10.2 ms |
| 6 | `ex6_stride4.cu` | STRIDE=4 + float4 | 向量化加载 + 更大 tile | TILE=16, STRIDE=4, BK=64, 32KB shared | 23.9 ms |
| 7 | `ex7_stride4_regtile.cu` | Register Tiling | shared→register 再分块 | TILE=16, STRIDE=4, BK=64, RK=4, 32KB shared | 5.2 ms |
| 8 | `ex8_reg_prefetch.cu` | Register Prefetching | warp tiling + 大 tile + 两层 prefetch | BM=128, BN=128, BK=8, block=256, 8KB shared | 3.83 ms |
| 9 | `ex9_double_buffer.cu` | Double Buffering | 双缓冲 shared + 省 __syncthreads | BM=128, BN=128, BK=8, block=256, 16KB shared | 3.60 ms |

累计加速比：**10.6×**（38.3ms → 3.60ms）

### 各版本说明

- **ex3 Naive**: 每个线程从 HBM 直读一行一列点积，无任何数据复用，访存瓶颈。
- **ex4 Tiled**: 引入 shared memory 分块，BK=32 时 28.6ms。BK=16 时反而慢（47.6ms），因为 barrier 开销太大（512 次迭代）。
- **ex5 Coarsening**: 每线程计算 STRIDE=2 个输出 + K-tile 扩大到 32，shared 仍 8KB，occupancy 满。
- **ex6 STRIDE=4**: STRIDE 扩到 4 → BK=64 → 32KB shared，occupancy 降到 4 block/SM，延迟暴露 → **反而比 ex5 慢**（23.9ms vs 10.2ms）。
- **ex7 Register Tiling**: 在 ex6 基础上加入 shared→register 分块（RK=4），shared 读减少 1/4，5.2ms。
- **ex8 Register Prefetching (Kernel 10)**: 完全重构——1D 线程映射、warp-level tiling（BM/BN/BK 解耦）、B 转置存储、两层 prefetch（Global→Reg 覆盖 500 cycle + Shared→Reg 乒乓覆盖 20 cycle）。8KB shared 极小，occupancy 高。
- **ex9 Double Buffering (Kernel 11)**: ex8 基础上加双缓冲 shared memory（16KB），读 buffer[cur] 和写 buffer[nxt] 同时进行，省掉一个 __syncthreads，3.60ms。

### 性能关键点

| 瓶颈 | ex3-6 受限于 | ex7-9 突破方式 |
|------|-------------|--------------|
| HBM 延迟（~500 cycle） | 串行等 | **Register Prefetching**: 先发射 LDG 再做计算 |
| Shared 延迟（~18 cycle） | 每迭代从 shared 读 | **寄存器乒乓**: 预取下一个 ik |
| __syncthreads（~20 cycle） | 每迭代 2 次 barrier | **Double Buffering**: 读/写不同 buffer, 省 1 次 |
| Occupancy（shared 占用） | BK=64 → 32KB → 4 block/SM | **大 tile + 小 BK**: BK=8 → 8KB → 高 occupancy |
| Bank conflict | 行主序 float4 加载潜在冲突 | **B 转置存储**: 零 bank conflict |

## 环境

- GPU: NVIDIA RTX 5080 (Blackwell, sm_120, 84 SM)
- CUDA Toolkit 13.3
- Windows + VS Build Tools + nvcc

## 编译运行

```powershell
# 一键编译全部 SGEMM (ex3-ex9):
cd gemm
powershell -Command "3..9 | ForEach-Object { nvcc ex$($_)_*.cu -o ex$($_).exe -Xcompiler '/utf-8' -arch=sm_120 ; if ($LASTEXITCODE -ne 0) { Write-Host 'FAIL' } else { .\ex$($_).exe } }"
cd ..

# 或逐条编译:
nvcc gemm/ex8_reg_prefetch.cu -o gemm/ex8_reg_prefetch.exe -Xcompiler "/utf-8" -arch=sm_120
.\gemm\ex8_reg_prefetch.exe
```

## 笔记

详细学习笔记（Obsidian 仓库）：SGEMM 优化原理、性能分析、GPU 架构知识。

- `obsidian-vault/cuda note/sgemm/` — 7 篇 SGEMM 笔记（Naive → Double Buffering）
