#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

__global__ void hgemm_wmma_naive(
    const half *A,   // M×K, row-major
    const half *B,   // K×N, row-major
    float     *C,    // M×N, row-major
    int M, int N, int K)
{
    // 每个 warp 负责 C 的一个 16×16 子块
    // warp 在 block 内按 threadIdx.x / 32 编号
    // grid.x 对应 M 方向，grid.y 对应 N 方向
    int warp_id  = threadIdx.x / 32;           // block 内第几个 warp
    int warp_m   = blockIdx.x * (blockDim.x / 32) + warp_id;  // M 方向的 warp 索引
    int warp_n   = blockIdx.y;                  // N 方向的 warp 索引

    int c_row = warp_m * WMMA_M;
    int c_col = warp_n * WMMA_N;

    if (c_row >= M || c_col >= N) return;

    // ---- 声明 fragment ----
    // 每个 fragment 是一个 warp 内所有线程共同持有的一块数据
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    // 累加器初始化为 0
    wmma::fill_fragment(c_frag, 0.0f);

    // ---- K 方向循环 ----
    // 每次迭代：从 HBM 加载 16×16 的 A 和 B tile，Tensor Core 一把算完
    for (int k = 0; k < K; k += WMMA_K) {
        // 从 global memory 加载 A 的 16×16 子块 (row-major)
        // 参数：fragment, 指针, leading dimension
        wmma::load_matrix_sync(a_frag, A + c_row * K + k, K);
        // 从 global memory 加载 B 的 16×16 子块 (col-major)
        // B 用 col_major → 按列加载，K 方向连续，正好是内积需要的方向
        wmma::load_matrix_sync(b_frag, B + k * N + c_col, N);
        // Tensor Core: C += A × B
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    // 写回 global memory
    wmma::store_matrix_sync(C + c_row * N + c_col, c_frag, N, wmma::mem_row_major);
}

// ============================================================
// 工具
// ============================================================
#define CUDA_CHK(call) do { \
    cudaError_t _e_ = (call); \
    if (_e_ != cudaSuccess) { \
        printf("CUDA ERROR at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e_)); \
        exit(1); \
    } \
} while(0)

// ============================================================
// main
// ============================================================
int main() {
    // 为了快速看到结果，先用小矩阵，跑通了再改成 4096
    const int M = 4096, N = 4096, K = 4096;

    printf("\n=== WMMA Naive HGEMM ===\n");
    printf("  M=N=K=%d,  WMMA tile: %dx%dx%d\n\n", M, WMMA_M, WMMA_N, WMMA_K);

    // ----- 初始化主机数据 -----
    half *hA_f16 = (half *)malloc(M * K * sizeof(half));
    half *hB_f16 = (half *)malloc(K * N * sizeof(half));
    for (int i = 0; i < M * K; i++) hA_f16[i] = __float2half((rand() % 100) / 100.0f);
    for (int i = 0; i < K * N; i++) hB_f16[i] = __float2half((rand() % 100) / 100.0f);

    // ----- GPU 显存 -----
    half *dA, *dB;
    float *dC;
    CUDA_CHK(cudaMalloc(&dA, M * K * sizeof(half)));
    CUDA_CHK(cudaMalloc(&dB, K * N * sizeof(half)));
    CUDA_CHK(cudaMalloc(&dC, M * N * sizeof(float)));
    CUDA_CHK(cudaMemcpy(dA, hA_f16, M * K * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHK(cudaMemcpy(dB, hB_f16, K * N * sizeof(half), cudaMemcpyHostToDevice));

    // ----- Launch 配置 -----
    // 每个 warp 算 16×16，每个 block 放 4 个 warp = 128 线程
    int warps_per_block = 4;
    dim3 block(32 * warps_per_block);   // 128 threads
    dim3 grid((M + WMMA_M - 1) / WMMA_M / warps_per_block,
              (N + WMMA_N - 1) / WMMA_N);

    printf("  Grid: (%d, %d),  Block: %d threads (%d warps)\n",
           grid.x, grid.y, block.x, warps_per_block);

    // ----- 计时 -----
    cudaEvent_t start, stop;
    CUDA_CHK(cudaEventCreate(&start));
    CUDA_CHK(cudaEventCreate(&stop));

    // 预热
    hgemm_wmma_naive<<<grid, block>>>(dA, dB, dC, M, N, K);
    CUDA_CHK(cudaDeviceSynchronize());

    // 正式计时
    CUDA_CHK(cudaEventRecord(start));
    hgemm_wmma_naive<<<grid, block>>>(dA, dB, dC, M, N, K);
    CUDA_CHK(cudaGetLastError());
    CUDA_CHK(cudaEventRecord(stop));
    CUDA_CHK(cudaEventSynchronize(stop));

    float ms;
    CUDA_CHK(cudaEventElapsedTime(&ms, start, stop));
    double tflops = 2.0 * M * N * K / (ms / 1000.0) / 1e12;
    printf("  Time: %.4f ms  |  %.2f TFLOPS\n", ms, tflops);

    // cleanup
    free(hA_f16); free(hB_f16);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return 0;
}
