#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "naive_attention.cuh"
#include "flashattention_v1.cuh"
#include "flashattention_v1_1.cuh"
#include "flashattention_v2.cuh"
#include "flashattn_minimal.cuh"

// ============================================================
// FlashAttention v1 Benchmark + 正确性验证
// ------------------------------------------------------------
// 一起对比的实现：
//   naive_attention.cuh      —— naive attention（golden reference）
//   flashattention_v1.cuh    —— 我们自己写的 flash attention v1
//   flashattention_v1_1.cuh  —— v1 + K/V cp.async 异步搬运 + 双缓冲
//   flashattention_v2.cuh    —— v2：Q 外层 + grid=(Q块,B*H) + 延迟归一化
//   flashattn_minimal.cuh    —— 参考实现（minimal 版，动态 shared，要求 Bc==Br）
// 本文件只负责：分配内存、launch、计时、对比正确性
//
// 内存布局：[B, H, N, D] 连续，索引 = ((b*H + h)*N + n)*D + d
// ============================================================

#define CUDA_CHK(call) do { \
    cudaError_t _e_ = (call); \
    if (_e_ != cudaSuccess) { \
        printf("CUDA ERROR at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e_)); \
        exit(1); \
    } \
} while(0)

#define HEAD_DIM 64   // head_dim
#define BR       32   // Q 块行数（= blockDim.x，每线程一行）
#define BC       32   // K/V 块行数（minimal 要求 BC == BR）

// ------------------------------------------------------------
// Attention 前向 FLOPs → TFLOPS
// 非 causal：S=QK^T 和 O=PV 各 2·N²·D，共 4·N²·D per (b,h)
// causal：  只算下三角，约一半 → 2·N²·D
// （softmax/exp 等非 GEMM 操作按惯例不计入）
// ------------------------------------------------------------
static double attn_tflops(int B, int H, int N, int D, bool causal, float ms)
{
    double flops = 4.0 * B * H * (double)N * N * D;
    if (causal) flops *= 0.5;
    return flops / (ms * 1e-3) / 1e12;
}

// ------------------------------------------------------------
// 初始化 flash 的状态量 l=0、m=-inf
// ------------------------------------------------------------
__global__ void init_state_kernel(float* l, float* m, int total) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) { l[i] = 0.0f; m[i] = -INFINITY; }
}

// ------------------------------------------------------------
// 以给定 (Br, Bc) 实例化我们的 flash kernel：预热 + 计时 + 对比 naive
// （模板参数必须编译期常量，所以扫参只能显式实例化几组）
// ------------------------------------------------------------
template<int Br, int Bc, int D>
void bench_flash(const float* dQ, const float* dK, const float* dV, float* dO,
                 float* dl, float* dm, int B, int H, int N, size_t n_state,
                 cudaEvent_t t0, cudaEvent_t t1,
                 const float* hO_naive, float* hO_buf, size_t n)
{
    dim3 grid(H, B);   // gridDim.x = H (head), gridDim.y = B (batch)

    init_state_kernel<<<(int)((n_state + 255) / 256), 256>>>(dl, dm, (int)n_state);
    CUDA_CHK(cudaDeviceSynchronize());
    flashattention_v1<Br, Bc, D><<<grid, Br>>>(dQ, dK, dV, dO, dl, dm, B, H, N);  // 预热
    CUDA_CHK(cudaGetLastError());
    CUDA_CHK(cudaDeviceSynchronize());

    init_state_kernel<<<(int)((n_state + 255) / 256), 256>>>(dl, dm, (int)n_state);
    CUDA_CHK(cudaDeviceSynchronize());

    CUDA_CHK(cudaEventRecord(t0));
    flashattention_v1<Br, Bc, D><<<grid, Br>>>(dQ, dK, dV, dO, dl, dm, B, H, N);
    CUDA_CHK(cudaGetLastError());
    CUDA_CHK(cudaEventRecord(t1));
    CUDA_CHK(cudaEventSynchronize(t1));
    float ms = 0.0f;
    CUDA_CHK(cudaEventElapsedTime(&ms, t0, t1));

    CUDA_CHK(cudaMemcpy(hO_buf, dO, n * sizeof(float), cudaMemcpyDeviceToHost));
    double diff = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double d = fabs((double)hO_naive[i] - (double)hO_buf[i]);
        if (d > diff) diff = d;
    }
    printf("    Br=%2d (block=%2d thr) | %9.3f ms | %7.2f TF | diff=%.2e\n",
           Br, Br, ms, attn_tflops(B, H, N, D, false, ms), diff);
}

// ------------------------------------------------------------
// v1.1（cp.async 双缓冲）：动态 shared，launch 前要解锁大小并传入字节数
// 布局 Ks[2][Bc][D] | Vs[2][Bc][D] | Qs[Br][D+1] | Os[Br][D+1]
// ------------------------------------------------------------
template<int Br, int Bc, int D>
void bench_flash_v11(const float* dQ, const float* dK, const float* dV, float* dO,
                     float* dl, float* dm, int B, int H, int N, size_t n_state,
                     cudaEvent_t t0, cudaEvent_t t1,
                     const float* hO_naive, float* hO_buf, size_t n)
{
    dim3 grid(H, B);
    size_t smem_bytes = (4 * (size_t)Bc * D + 2 * (size_t)Br * (D + 1)) * sizeof(float);

    // 超过 48KB 的动态 shared 必须先解锁（可解锁到 ~99KB）
    CUDA_CHK(cudaFuncSetAttribute(flashattention_v1_1<Br, Bc, D>,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  (int)smem_bytes));

    init_state_kernel<<<(int)((n_state + 255) / 256), 256>>>(dl, dm, (int)n_state);
    CUDA_CHK(cudaDeviceSynchronize());
    flashattention_v1_1<Br, Bc, D><<<grid, Br, smem_bytes>>>(dQ, dK, dV, dO, dl, dm, B, H, N);  // 预热
    CUDA_CHK(cudaGetLastError());
    CUDA_CHK(cudaDeviceSynchronize());

    init_state_kernel<<<(int)((n_state + 255) / 256), 256>>>(dl, dm, (int)n_state);
    CUDA_CHK(cudaDeviceSynchronize());

    CUDA_CHK(cudaEventRecord(t0));
    flashattention_v1_1<Br, Bc, D><<<grid, Br, smem_bytes>>>(dQ, dK, dV, dO, dl, dm, B, H, N);
    CUDA_CHK(cudaGetLastError());
    CUDA_CHK(cudaEventRecord(t1));
    CUDA_CHK(cudaEventSynchronize(t1));
    float ms = 0.0f;
    CUDA_CHK(cudaEventElapsedTime(&ms, t0, t1));

    CUDA_CHK(cudaMemcpy(hO_buf, dO, n * sizeof(float), cudaMemcpyDeviceToHost));
    double diff = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double d = fabs((double)hO_naive[i] - (double)hO_buf[i]);
        if (d > diff) diff = d;
    }
    printf("    Br=%2d (block=%2d thr, smem=%5.1fKB) | %9.3f ms | %7.2f TF | diff=%.2e\n",
           Br, Br, smem_bytes / 1024.0, ms, attn_tflops(B, H, N, D, false, ms), diff);
}

// ------------------------------------------------------------
// v2.0（标量版）：Q 外层、grid=(Q块数,B*H)，无需 init_state（m/l 片上初始化）
// CAUSAL=true 时启用因果 mask（块级跳过 + 对角线块逐元素 mask）
// shared 布局 Ks[2][Bc][D] | Vs[2][Bc][D] | Qs[Br][D+1] | Oacc[Br][D+1]
// hO_ref 传「同模式的 golden」：非 causal 对 naive，causal 对 naive causal
// ------------------------------------------------------------
template<bool CAUSAL, int Br, int Bc, int D>
void bench_flash_v2(const float* dQ, const float* dK, const float* dV, float* dO,
                    int B, int H, int N,
                    cudaEvent_t t0, cudaEvent_t t1,
                    const float* hO_ref, float* hO_buf, size_t n)
{
    dim3 grid((N + Br - 1) / Br, B * H);   // gridDim.x = Q 块数，gridDim.y = B*H
    size_t smem_bytes = (4 * (size_t)Bc * D + 2 * (size_t)Br * (D + 1)) * sizeof(float);

    CUDA_CHK(cudaFuncSetAttribute(flashattention_v2<CAUSAL, Br, Bc, D>,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  (int)smem_bytes));

    flashattention_v2<CAUSAL, Br, Bc, D><<<grid, Br, smem_bytes>>>(dQ, dK, dV, dO, N);  // 预热
    CUDA_CHK(cudaGetLastError());
    CUDA_CHK(cudaDeviceSynchronize());

    CUDA_CHK(cudaEventRecord(t0));
    flashattention_v2<CAUSAL, Br, Bc, D><<<grid, Br, smem_bytes>>>(dQ, dK, dV, dO, N);
    CUDA_CHK(cudaGetLastError());
    CUDA_CHK(cudaEventRecord(t1));
    CUDA_CHK(cudaEventSynchronize(t1));
    float ms = 0.0f;
    CUDA_CHK(cudaEventElapsedTime(&ms, t0, t1));

    CUDA_CHK(cudaMemcpy(hO_buf, dO, n * sizeof(float), cudaMemcpyDeviceToHost));
    double diff = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double d = fabs((double)hO_ref[i] - (double)hO_buf[i]);
        if (d > diff) diff = d;
    }
    printf("    Br=%3d (grid=%4d blk, block=%3d thr, smem=%5.1fKB) | %9.3f ms | %7.2f TF | diff=%.2e\n",
           Br, grid.x, Br, smem_bytes / 1024.0, ms, attn_tflops(B, H, N, D, CAUSAL, ms), diff);
}

// ------------------------------------------------------------
// 对单个 N：跑 naive / flash / minimal，对比正确性 + 计时
// ------------------------------------------------------------
void run_case(int B, int H, int N)
{
    size_t n       = (size_t)B * H * N * HEAD_DIM;
    size_t n_state = (size_t)B * H * N;

    // ---- host 数据 ----
    float* hQ = (float*)malloc(n * sizeof(float));
    float* hK = (float*)malloc(n * sizeof(float));
    float* hV = (float*)malloc(n * sizeof(float));
    float* hO_naive   = (float*)malloc(n * sizeof(float));
    float* hO_flash   = (float*)malloc(n * sizeof(float));
    float* hO_minimal = (float*)malloc(n * sizeof(float));
    for (size_t i = 0; i < n; ++i) {
        hQ[i] = (float)(rand() % 1000) / 1000.0f - 0.5f;
        hK[i] = (float)(rand() % 1000) / 1000.0f - 0.5f;
        hV[i] = (float)(rand() % 1000) / 1000.0f - 0.5f;
    }

    // ---- device 数据 ----
    float *dQ, *dK, *dV, *dO_naive, *dO_flash, *dO_minimal, *dl, *dm;
    CUDA_CHK(cudaMalloc(&dQ, n * sizeof(float)));
    CUDA_CHK(cudaMalloc(&dK, n * sizeof(float)));
    CUDA_CHK(cudaMalloc(&dV, n * sizeof(float)));
    CUDA_CHK(cudaMalloc(&dO_naive, n * sizeof(float)));
    CUDA_CHK(cudaMalloc(&dO_flash, n * sizeof(float)));
    CUDA_CHK(cudaMalloc(&dO_minimal, n * sizeof(float)));
    CUDA_CHK(cudaMalloc(&dl, n_state * sizeof(float)));
    CUDA_CHK(cudaMalloc(&dm, n_state * sizeof(float)));

    CUDA_CHK(cudaMemcpy(dQ, hQ, n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHK(cudaMemcpy(dK, hK, n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHK(cudaMemcpy(dV, hV, n * sizeof(float), cudaMemcpyHostToDevice));

    cudaEvent_t t0, t1;
    CUDA_CHK(cudaEventCreate(&t0));
    CUDA_CHK(cudaEventCreate(&t1));

    // ---- naive（golden）----
    int total_threads = B * H * N;
    int block_naive = 256;
    int grid_naive  = (total_threads + block_naive - 1) / block_naive;

    naive_attention_kernel<false, HEAD_DIM><<<grid_naive, block_naive>>>(dQ, dK, dV, dO_naive, B, H, N);  // 预热
    CUDA_CHK(cudaDeviceSynchronize());

    CUDA_CHK(cudaEventRecord(t0));
    naive_attention_kernel<false, HEAD_DIM><<<grid_naive, block_naive>>>(dQ, dK, dV, dO_naive, B, H, N);
    CUDA_CHK(cudaEventRecord(t1));
    CUDA_CHK(cudaEventSynchronize(t1));
    float ms_naive = 0.0f;
    CUDA_CHK(cudaEventElapsedTime(&ms_naive, t0, t1));

    // ---- flash attention v1（我们自己写的）----
    dim3 grid_flash(H, B);   // 我们的 flash：gridDim.x = H (head), gridDim.y = B (batch)
    int  block_flash = BR;

    init_state_kernel<<<(int)((n_state + 255) / 256), 256>>>(dl, dm, (int)n_state);
    CUDA_CHK(cudaDeviceSynchronize());

    flashattention_v1<BR, BC, HEAD_DIM><<<grid_flash, block_flash>>>(dQ, dK, dV, dO_flash, dl, dm, B, H, N);  // 预热
    CUDA_CHK(cudaGetLastError());
    CUDA_CHK(cudaDeviceSynchronize());

    init_state_kernel<<<(int)((n_state + 255) / 256), 256>>>(dl, dm, (int)n_state);
    CUDA_CHK(cudaDeviceSynchronize());

    CUDA_CHK(cudaEventRecord(t0));
    flashattention_v1<BR, BC, HEAD_DIM><<<grid_flash, block_flash>>>(dQ, dK, dV, dO_flash, dl, dm, B, H, N);
    CUDA_CHK(cudaGetLastError());
    CUDA_CHK(cudaEventRecord(t1));
    CUDA_CHK(cudaEventSynchronize(t1));
    float ms_flash = 0.0f;
    CUDA_CHK(cudaEventElapsedTime(&ms_flash, t0, t1));

    // ---- flash attention minimal（参考实现）----
    // 注意：minimal 的 grid 方向是 gridDim.x = batch, gridDim.y = head，和我们的 flash 相反
    int Tc = N / BC;
    int Tr = N / BR;
    float softmax_scale = 1.0f / sqrtf((float)HEAD_DIM);
    size_t shared_bytes = (3 * (size_t)BC * HEAD_DIM + (size_t)BC * BR) * sizeof(float);  // Qi + Kj + Vj + S

    init_state_kernel<<<(int)((n_state + 255) / 256), 256>>>(dl, dm, (int)n_state);
    CUDA_CHK(cudaDeviceSynchronize());

    flashAttentionMinimal<<<dim3(B, H), BR, shared_bytes>>>(dQ, dK, dV, B, H, N, HEAD_DIM, Tc, Tr, BC, BR, softmax_scale, dl, dm, dO_minimal);  // 预热
    CUDA_CHK(cudaGetLastError());
    CUDA_CHK(cudaDeviceSynchronize());

    init_state_kernel<<<(int)((n_state + 255) / 256), 256>>>(dl, dm, (int)n_state);
    CUDA_CHK(cudaDeviceSynchronize());

    CUDA_CHK(cudaEventRecord(t0));
    flashAttentionMinimal<<<dim3(B, H), BR, shared_bytes>>>(dQ, dK, dV, B, H, N, HEAD_DIM, Tc, Tr, BC, BR, softmax_scale, dl, dm, dO_minimal);
    CUDA_CHK(cudaGetLastError());
    CUDA_CHK(cudaEventRecord(t1));
    CUDA_CHK(cudaEventSynchronize(t1));
    float ms_minimal = 0.0f;
    CUDA_CHK(cudaEventElapsedTime(&ms_minimal, t0, t1));

    // ---- 正确性对比 ----
    CUDA_CHK(cudaMemcpy(hO_naive, dO_naive, n * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHK(cudaMemcpy(hO_flash, dO_flash, n * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHK(cudaMemcpy(hO_minimal, dO_minimal, n * sizeof(float), cudaMemcpyDeviceToHost));

    double diff_flash = 0.0, diff_minimal = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double d1 = fabs((double)hO_naive[i] - (double)hO_flash[i]);
        double d2 = fabs((double)hO_naive[i] - (double)hO_minimal[i]);
        if (d1 > diff_flash)   diff_flash   = d1;
        if (d2 > diff_minimal) diff_minimal = d2;
    }
    const char* ok = (diff_flash < 1e-2 && diff_minimal < 1e-2) ? "PASS" : "FAIL";

    printf("  N=%5d | naive=%9.3f ms (%6.2f TF) | flash=%9.3f ms (%6.2f TF) | minimal=%9.3f ms (%6.2f TF)\n",
           N, ms_naive, attn_tflops(B, H, N, HEAD_DIM, false, ms_naive),
           ms_flash, attn_tflops(B, H, N, HEAD_DIM, false, ms_flash),
           ms_minimal, attn_tflops(B, H, N, HEAD_DIM, false, ms_minimal));
    printf("          | diff(flash)=%.2e diff(min)=%.2e  [%s]\n", diff_flash, diff_minimal, ok);

    // ---- Br 扫参（固定 Bc=32）----
    // 受 48KB 静态 shared 限制，Bc=32/D=64 时 Br 最大只能到 48：
    //   (2*48*65 + 2*32*64) * 4B ≈ 41KB ✓     (2*64*65 + 2*32*64) * 4B ≈ 49.7KB ✗
    printf("    -- Br sweep (Bc=%d fixed) --\n", BC);
    bench_flash<16, BC, HEAD_DIM>(dQ, dK, dV, dO_flash, dl, dm, B, H, N, n_state, t0, t1, hO_naive, hO_flash, n);
    bench_flash<32, BC, HEAD_DIM>(dQ, dK, dV, dO_flash, dl, dm, B, H, N, n_state, t0, t1, hO_naive, hO_flash, n);
    bench_flash<48, BC, HEAD_DIM>(dQ, dK, dV, dO_flash, dl, dm, B, H, N, n_state, t0, t1, hO_naive, hO_flash, n);

    // ---- v1.1 Br 扫参（cp.async 双缓冲；动态 shared 解锁后 Br 不再受 48KB 静态上限约束）----
    printf("    -- v1.1 Br sweep (Bc=%d fixed, cp.async + double buffer) --\n", BC);
    bench_flash_v11<16, BC, HEAD_DIM>(dQ, dK, dV, dO_flash, dl, dm, B, H, N, n_state, t0, t1, hO_naive, hO_flash, n);
    bench_flash_v11<32, BC, HEAD_DIM>(dQ, dK, dV, dO_flash, dl, dm, B, H, N, n_state, t0, t1, hO_naive, hO_flash, n);
    bench_flash_v11<48, BC, HEAD_DIM>(dQ, dK, dV, dO_flash, dl, dm, B, H, N, n_state, t0, t1, hO_naive, hO_flash, n);
    bench_flash_v11<64, BC, HEAD_DIM>(dQ, dK, dV, dO_flash, dl, dm, B, H, N, n_state, t0, t1, hO_naive, hO_flash, n);

    // ---- v2.0 Br 扫参（Q 外层；Br=128 时 4 warp 沿 Q 切，smem≈97KB 贴 99KB 上限）----
    printf("    -- v2.0 Br sweep (Bc=%d fixed, Q-outer grid) --\n", BC);
    bench_flash_v2<false, 32,  BC, HEAD_DIM>(dQ, dK, dV, dO_flash, B, H, N, t0, t1, hO_naive, hO_flash, n);
    bench_flash_v2<false, 64,  BC, HEAD_DIM>(dQ, dK, dV, dO_flash, B, H, N, t0, t1, hO_naive, hO_flash, n);
    bench_flash_v2<false, 96,  BC, HEAD_DIM>(dQ, dK, dV, dO_flash, B, H, N, t0, t1, hO_naive, hO_flash, n);
    bench_flash_v2<false, 128, BC, HEAD_DIM>(dQ, dK, dV, dO_flash, B, H, N, t0, t1, hO_naive, hO_flash, n);

    // ---- causal 对比：naive causal 做 golden，v2 causal 测块级跳过的收益 ----
    naive_attention_kernel<true, HEAD_DIM><<<grid_naive, block_naive>>>(dQ, dK, dV, dO_naive, B, H, N);
    CUDA_CHK(cudaDeviceSynchronize());
    float ms_naive_causal = 0.0f;
    CUDA_CHK(cudaEventRecord(t0));
    naive_attention_kernel<true, HEAD_DIM><<<grid_naive, block_naive>>>(dQ, dK, dV, dO_naive, B, H, N);
    CUDA_CHK(cudaEventRecord(t1));
    CUDA_CHK(cudaEventSynchronize(t1));
    CUDA_CHK(cudaEventElapsedTime(&ms_naive_causal, t0, t1));
    CUDA_CHK(cudaMemcpy(hO_naive, dO_naive, n * sizeof(float), cudaMemcpyDeviceToHost));
    printf("    naive causal (golden)                        | %9.3f ms | %7.2f TF\n",
           ms_naive_causal, attn_tflops(B, H, N, HEAD_DIM, true, ms_naive_causal));

    printf("    -- v2.0 CAUSAL sweep (块级跳过 + 对角线块 mask) --\n");
    bench_flash_v2<true, 32,  BC, HEAD_DIM>(dQ, dK, dV, dO_flash, B, H, N, t0, t1, hO_naive, hO_flash, n);
    bench_flash_v2<true, 64,  BC, HEAD_DIM>(dQ, dK, dV, dO_flash, B, H, N, t0, t1, hO_naive, hO_flash, n);
    bench_flash_v2<true, 96,  BC, HEAD_DIM>(dQ, dK, dV, dO_flash, B, H, N, t0, t1, hO_naive, hO_flash, n);
    bench_flash_v2<true, 128, BC, HEAD_DIM>(dQ, dK, dV, dO_flash, B, H, N, t0, t1, hO_naive, hO_flash, n);

    // ---- 清理 ----
    free(hQ); free(hK); free(hV); free(hO_naive); free(hO_flash); free(hO_minimal);
    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO_naive); cudaFree(dO_flash); cudaFree(dO_minimal);
    cudaFree(dl); cudaFree(dm);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
}

int main()
{
    const int B = 4;
    const int H = 8;

    printf("\n=== FlashAttention v1 Benchmark & Correctness ===\n");
    printf("  B=%d  H=%d  D=%d  Br=%d  Bc=%d  (FP32)\n", B, H, HEAD_DIM, BR, BC);
    printf("  (flash/minimal 用 __expf 快速指数，与 naive 的 expf 有微小精度差异)\n\n");

    const int Ns[] = {1024, 2048, 4096, 8192};
    for (int i = 0; i < (int)(sizeof(Ns) / sizeof(Ns[0])); ++i) {
        run_case(B, H, Ns[i]);
    }

    printf("\n  Done.\n\n");
    return 0;
}
