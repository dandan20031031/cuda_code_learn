# CUDA 编程学习

个人 CUDA 编程学习代码，从零开始掌握 GPU 并行计算。

## 项目目录

```
cuda_code/
├── sgemm/                        # SGEMM 优化演进线（ex0-ex9）
│   ├── ex0_cublas.cu
│   ├── ex1.cu
│   ├── ex2.cu
│   ├── ex3_sgemm_naive.cu
│   ├── ex4_sgemm_tiled.cu
│   ├── ex5_coarsen.cu
│   ├── ex6_stride4.cu
│   ├── ex7_stride4_regtile.cu
│   ├── ex8_reg_prefetch.cu
│   └── ex9_double_buffer.cu
├── hgemm/                        # WMMA HGEMM 优化演进线（v0-v8）
│   ├── hgemm_v0_cublas.cu        # cuBLAS 参考
│   ├── hgemm_wmma_naivev1.cu     # Naive WMMA
│   ├── hgemm_v2.cu               # Shared Tiling
│   ├── hgemm_v3_coarsen.cu       # Warp Coarsening
│   ├── hgemm_v4_double_buffer.cu # cp.async + Double Buffer
│   ├── hgemm_v5_pad.cu           # Padding（消 bank conflict）
│   ├── hgemm_v6_transpose.cu     # B 转置（失败版本）
│   ├── hgemm_v7_tile128.cu       # 大 tile 128×128
│   └── hgemm_v8_check.cu         # half 累加器（精度校验）
├── softmax/                      # Safe Softmax 优化演进线（v1-v5）
│   ├── softmax_bench.cu          # benchmark 主程序
│   ├── softmax_v1_naive.cuh      # Naive（单线程）
│   ├── softmax_v2_sharedreduce.cuh   # Shared Memory 归约
│   ├── softmax_v3_warpshuffle.cuh    # Warp Shuffle 归约
│   ├── softmax_v4_float4.cuh     # float4 向量化
│   ├── softmax_v5_twopass.cuh    # Online 两遍融合
│   └── online-softmax/           # Online Softmax 优化演进线（v0-v4）
│       ├── online_softmax_bench.cu       # benchmark 主程序
│       ├── online_softmax_v0_naive.cuh   # 单线程 Online
│       ├── online_softmax_v1_warpshuffle.cuh  # Block 并行 + Warp Shuffle
│       ├── online_softmax_v2_float4.cuh       # float4 向量化
│       ├── online_softmax_v3_regcache.cuh     # 寄存器缓存 one-pass
│       └── online_softmax_v4_gridstride.cuh   # Grid Stride
├── cublass_gemm/                 # cuBLAS GEMM 参考
│   └── cublas.cpp
├── .vscode/                      # 编译任务配置
│   └── tasks.json
└── README.md
```
