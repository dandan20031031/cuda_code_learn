#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

// Tensor Core tile: warp 级别，固定 16×16×16
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// Block tile: shared memory 级别，你来定
#define BM 64
#define BN 64
#define BK 32

// 每个 warp 多算几个 tile：用更少的 warp 干更多活
#define WARPS_M 2                // M 方向 2 个 warp（原来是 4）
#define WARPS_N 2                // N 方向 2 个 warp（原来是 4）
#define WARPS   (WARPS_M * WARPS_N) // 4 warps（原来是 16）

#define COARSE_M (BM / WMMA_M / WARPS_M)   // 64/16/2 = 2，每 warp 沿 M 算 2 个 tile
#define COARSE_N (BN / WMMA_N / WARPS_N)   // 64/16/2 = 2，每 warp 沿 N 算 2 个 tile

__global__ void hgemm_wmma_coarsen(
    const half *A,
    const half *B,
    float       *C,
    int M, int N, int K)
{
    int warp_id = threadIdx.x / 32;
    int warp_m  = warp_id / WARPS_N;     // 0~1
    int warp_n  = warp_id % WARPS_N;     // 0~1

    // 这个 warp 管 4 个 tile，起始位置
    int c_row_base = blockIdx.x * BM + warp_m * COARSE_M * WMMA_M;
    int c_col_base = blockIdx.y * BN + warp_n * COARSE_N * WMMA_N;
    if (c_row_base >= M || c_col_base >= N) return;

    // ---- shared memory ----
    __shared__ half As[BM][BK];
    __shared__ half Bs[BK][BN];

    // ---- 多个累加器：c_frag[mi][ni] 对应 4 个输出 tile ----
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag[COARSE_M][COARSE_N];
    for (int mi = 0; mi < COARSE_M; mi++)
        for (int ni = 0; ni < COARSE_N; ni++)
            wmma::fill_fragment(c_frag[mi][ni], 0.0f);

    // ---- K 方向大循环 ----
    for (int kb = 0; kb < K; kb += BK) {

        // ===== 步骤1: 搬 HBM → shared（128 线程，每人搬 BM×BK/blocksize 个 half）=====
        for (int i = threadIdx.x; i < BM * BK; i += blockDim.x) {
            int row = i / BK;
            int col = i % BK;
            int gr = blockIdx.x * BM + row;
            int gc = kb + col;
            As[row][col] = (gr < M && gc < K) ? A[gr * K + gc] : __float2half(0.0f);
        }
        for (int i = threadIdx.x; i < BK * BN; i += blockDim.x) {
            int row = i / BN;
            int col = i % BN;
            int gr = kb + row;
            int gc = blockIdx.y * BN + col;
            Bs[row][col] = (gr < K && gc < N) ? B[gr * N + gc] : __float2half(0.0f);
        }
        __syncthreads();

        // ===== 步骤2+3: 一个 warp 串行算 4 个 tile =====
        for (int mi = 0; mi < COARSE_M; mi++) {
            for (int ni = 0; ni < COARSE_N; ni++) {
                int a_off = (warp_m * COARSE_M + mi) * WMMA_M;
                int b_off = (warp_n * COARSE_N + ni) * WMMA_N;
                wmma::load_matrix_sync(a_frag, &As[a_off][0], BK);
                wmma::load_matrix_sync(b_frag, &Bs[0][b_off], BN);
                wmma::mma_sync(c_frag[mi][ni], a_frag, b_frag, c_frag[mi][ni]);
            }
        }
        __syncthreads();
    }

    // 写回 global memory
    for (int mi = 0; mi < COARSE_M; mi++) {
        for (int ni = 0; ni < COARSE_N; ni++) {
            int cr = c_row_base + mi * WMMA_M;
            int cc = c_col_base + ni * WMMA_N;
            if (cr < M && cc < N)
                wmma::store_matrix_sync(C + cr * N + cc, c_frag[mi][ni], N, wmma::mem_row_major);
        }
    }
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

    printf("\n=== WMMA Coarsening HGEMM ===\n");
    printf("  M=N=K=%d\n", M);
    printf("  Block tile: BM=%d BN=%d BK=%d\n", BM, BN, BK);
    printf("  Warps: %dx%d=%d  (coarsening: %dx%d tiles/warp)\n",
           WARPS_M, WARPS_N, WARPS, COARSE_M, COARSE_N);
    printf("  Block: %d threads\n", 32 * WARPS);

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
    dim3 block(32 * WARPS);               // 128 threads
    dim3 grid((M + BM - 1) / BM,          // M 方向 block 数
              (N + BN - 1) / BN);         // N 方向 block 数

    printf("  Grid: (%d, %d)\n\n", grid.x, grid.y);

    // ----- 计时 -----
    cudaEvent_t start, stop;
    CUDA_CHK(cudaEventCreate(&start));
    CUDA_CHK(cudaEventCreate(&stop));

    // 预热
    hgemm_wmma_coarsen<<<grid, block>>>(dA, dB, dC, M, N, K);
    CUDA_CHK(cudaDeviceSynchronize());

    // 正式计时
    CUDA_CHK(cudaEventRecord(start));
    hgemm_wmma_coarsen<<<grid, block>>>(dA, dB, dC, M, N, K);
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
