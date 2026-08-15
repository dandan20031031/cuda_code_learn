#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

// Tensor Core tile: warp ，固定 16×16×16
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// Block tile: shared memory
#define BM 64
#define BN 64
#define BK 32

#define WARPS_M (BM / WMMA_M)   // 4
#define WARPS_N (BN / WMMA_N)   // 4
#define WARPS   (WARPS_M * WARPS_N) // 16

__global__ void hgemm_wmma_tiled(
    const half *A,
    const half *B,
    float       *C,
    int M, int N, int K)
{
    // ---- warp 坐标（二维）----
    
    int warp_id = threadIdx.x / 32;
    int warp_m  = warp_id / WARPS_N;  // 0~3, M 方向
    int warp_n  = warp_id % WARPS_N;  // 0~3, N 方向

    // 当前 warp 负责的 C 子块起始位置
    int c_row = blockIdx.x * BM + warp_m * WMMA_M;
    int c_col = blockIdx.y * BN + warp_n * WMMA_N;
    if (c_row >= M || c_col >= N) return;

    // ---- shared memory（block 内所有 warp 共享）----
    __shared__ half As[BM][BK];       // BM×BK
    __shared__ half Bs[BK][BN];       // BK×BN, 注意 B 存 K×N

    // ---- fragment（warp ）----
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    // ---- K 方向大循环 ----
    for (int kb = 0; kb < K; kb += BK) {

        // ===== 步骤1: block 所有线程协作，从 HBM 搬到 shared =====
        // A: 每线程搬 (BM*BK) / (32*WARPS) = (64*32)/512 = 4 个 half
        for (int i = threadIdx.x; i < BM * BK; i += blockDim.x) {
            int row = i / BK;           // shared A 的行
            int col = i % BK;           // shared A 的列
            int global_row = blockIdx.x * BM + row;
            int global_col = kb + col;
            As[row][col] = (global_row < M && global_col < K)
                         ? A[global_row * K + global_col] : __float2half(0.0f);
        }
        // B: 每线程搬 (BK*BN) / (32*WARPS) = (32*64)/512 = 4 个 half
        for (int i = threadIdx.x; i < BK * BN; i += blockDim.x) {
            int row = i / BN;           // shared B 的行 = K 方向
            int col = i % BN;           // shared B 的列 = N 方向
            int global_row = kb + row;
            int global_col = blockIdx.y * BN + col;
            Bs[row][col] = (global_row < K && global_col < N)
                         ? B[global_row * N + global_col] : __float2half(0.0f);
        }
        __syncthreads();    // 确保所有线程搬完

        // ===== 步骤2: 每个 warp 从 shared 加载自己的 16×16 fragment =====
        // A fragment: 从 As[warp_m*16][0] 开始，连续 16 行 16 列
        wmma::load_matrix_sync(a_frag, &As[warp_m * WMMA_M][0], BK);
        // B fragment: 从 Bs[0][warp_n*16] 开始
        wmma::load_matrix_sync(b_frag, &Bs[0][warp_n * WMMA_N], BN);

        // ===== 步骤3: Tensor Core 计算 =====
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

        __syncthreads();    // 确保所有 warp 用完 shared 再写下一轮
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
    const int M = 4096, N = 4096, K = 4096;

    printf("\n=== WMMA Tiled HGEMM ===\n");
    printf("  M=N=K=%d\n", M);
    printf("  Block tile: BM=%d BN=%d BK=%d\n", BM, BN, BK);
    printf("  Warps: %dx%d=%d,  Block: %d threads\n", WARPS_M, WARPS_N, WARPS, 32 * WARPS);

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
    dim3 block(32 * WARPS);               // 512 threads
    dim3 grid((M + BM - 1) / BM,          // M 方向 block 数
              (N + BN - 1) / BN);         // N 方向 block 数

    printf("  Grid: (%d, %d)\n\n", grid.x, grid.y);

    // ----- 计时 -----
    cudaEvent_t start, stop;
    CUDA_CHK(cudaEventCreate(&start));
    CUDA_CHK(cudaEventCreate(&stop));

    // 预热
    hgemm_wmma_tiled<<<grid, block>>>(dA, dB, dC, M, N, K);
    CUDA_CHK(cudaDeviceSynchronize());

    // 正式计时
    CUDA_CHK(cudaEventRecord(start));
    hgemm_wmma_tiled<<<grid, block>>>(dA, dB, dC, M, N, K);
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
