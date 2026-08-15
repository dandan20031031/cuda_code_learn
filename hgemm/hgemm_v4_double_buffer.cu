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

// Block tile: shared memory 级别
#define BM 64
#define BN 64
#define BK 32

#define WARPS_M 2
#define WARPS_N 2
#define WARPS   (WARPS_M * WARPS_N)      // 4 warps

#define COARSE_M (BM / WMMA_M / WARPS_M) // 2
#define COARSE_N (BN / WMMA_N / WARPS_N) // 2

// ---- cp.async 内联 PTX 封装 ----
// 16 字节异步拷贝：HBM → shared，线程不阻塞
__device__ __forceinline__ void cp_async16(uint32_t smem_dst, const void *gmem_src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                 :: "r"(smem_dst), "l"(gmem_src));
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n");
}

template <int N>
__device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}

// ---- 异步搬 A tile: HBM → shared，每线程搬 2 个 16 字节块 ----
__device__ __forceinline__ void load_As_async(half As[BM][BK], const half *A,
                                              int K, int m_off, int kb)
{
    uint32_t smem = (uint32_t)__cvta_generic_to_shared(As);
    constexpr int CHUNKS_PER_ROW = BK / 8;        // 32/8 = 4 块/行
    constexpr int TOTAL_CHUNKS   = BM * CHUNKS_PER_ROW; // 256 块
    #pragma unroll
    for (int c = threadIdx.x; c < TOTAL_CHUNKS; c += blockDim.x) {
        int row  = c / CHUNKS_PER_ROW;
        int col8 = c % CHUNKS_PER_ROW;             // 8 个 half 为单位
        uint32_t dst = smem + (row * BK + col8 * 8) * sizeof(half);
        const half *src = &A[(m_off + row) * K + kb + col8 * 8];
        cp_async16(dst, src);
    }
}

// ---- 异步搬 B tile ----
__device__ __forceinline__ void load_Bs_async(half Bs[BK][BN], const half *B,
                                              int N, int n_off, int kb)
{
    uint32_t smem = (uint32_t)__cvta_generic_to_shared(Bs);
    constexpr int CHUNKS_PER_ROW = BN / 8;        // 64/8 = 8 块/行
    constexpr int TOTAL_CHUNKS   = BK * CHUNKS_PER_ROW; // 256 块
    #pragma unroll
    for (int c = threadIdx.x; c < TOTAL_CHUNKS; c += blockDim.x) {
        int row  = c / CHUNKS_PER_ROW;
        int col8 = c % CHUNKS_PER_ROW;
        uint32_t dst = smem + (row * BN + col8 * 8) * sizeof(half);
        const half *src = &B[(kb + row) * N + n_off + col8 * 8];
        cp_async16(dst, src);
    }
}

__global__ void hgemm_wmma_async(
    const half *A,
    const half *B,
    float       *C,
    int M, int N, int K)
{
    int warp_id = threadIdx.x / 32;
    int warp_m  = warp_id / WARPS_N;
    int warp_n  = warp_id % WARPS_N;

    int c_row_base = blockIdx.x * BM + warp_m * COARSE_M * WMMA_M;
    int c_col_base = blockIdx.y * BN + warp_n * COARSE_N * WMMA_N;
    if (c_row_base >= M || c_col_base >= N) return;

    // 双 buffer shared memory
    __shared__ half As[2][BM][BK];
    __shared__ half Bs[2][BK][BN];

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag[COARSE_M][COARSE_N];
    #pragma unroll
    for (int mi = 0; mi < COARSE_M; mi++)
        #pragma unroll
        for (int ni = 0; ni < COARSE_N; ni++)
            wmma::fill_fragment(c_frag[mi][ni], 0.0f);

    // 预取第 0 个 K tile 到 buffer[0]，commit 成 group 0
    load_As_async(As[0], A, K, blockIdx.x * BM, 0);
    load_Bs_async(Bs[0], B, N, blockIdx.y * BN, 0);
    cp_async_commit();

    const int n_iters = K / BK;
    for (int it = 0; it < n_iters; it++) {
        int cur = it & 1;
        int nxt = cur ^ 1;

        // 预取下一个 K tile（异步，不阻塞）
        if (it + 1 < n_iters) {
            int next_kb = (it + 1) * BK;
            load_As_async(As[nxt], A, K, blockIdx.x * BM, next_kb);
            load_Bs_async(Bs[nxt], B, N, blockIdx.y * BN, next_kb);
            cp_async_commit();
        }

        // 等当前 tile 的数据到位（允许下一 tile 的拷贝仍在后台进行）
        cp_async_wait<1>();
        __syncthreads();

        // ---- 计算当前 buffer：外层 4 个 tile，内层 K 子块循环（修复缺 K 的 bug）----
        #pragma unroll
        for (int mi = 0; mi < COARSE_M; mi++) {
            #pragma unroll
            for (int ni = 0; ni < COARSE_N; ni++) {
                int a_off = (warp_m * COARSE_M + mi) * WMMA_M;
                int b_off = (warp_n * COARSE_N + ni) * WMMA_N;
                #pragma unroll
                for (int kk = 0; kk < BK; kk += WMMA_K) {
                    wmma::load_matrix_sync(a_frag, &As[cur][a_off][kk], BK);
                    wmma::load_matrix_sync(b_frag, &Bs[cur][kk][b_off], BN);
                    wmma::mma_sync(c_frag[mi][ni], a_frag, b_frag, c_frag[mi][ni]);
                }
            }
        }
        __syncthreads();
    }

    // 写回 global memory，手动展开
    if (c_row_base + 0 * WMMA_M < M && c_col_base + 0 * WMMA_N < N)
        wmma::store_matrix_sync(C + (c_row_base + 0 * WMMA_M) * N + (c_col_base + 0 * WMMA_N), c_frag[0][0], N, wmma::mem_row_major);
    if (c_row_base + 0 * WMMA_M < M && c_col_base + 1 * WMMA_N < N)
        wmma::store_matrix_sync(C + (c_row_base + 0 * WMMA_M) * N + (c_col_base + 1 * WMMA_N), c_frag[0][1], N, wmma::mem_row_major);
    if (c_row_base + 1 * WMMA_M < M && c_col_base + 0 * WMMA_N < N)
        wmma::store_matrix_sync(C + (c_row_base + 1 * WMMA_M) * N + (c_col_base + 0 * WMMA_N), c_frag[1][0], N, wmma::mem_row_major);
    if (c_row_base + 1 * WMMA_M < M && c_col_base + 1 * WMMA_N < N)
        wmma::store_matrix_sync(C + (c_row_base + 1 * WMMA_M) * N + (c_col_base + 1 * WMMA_N), c_frag[1][1], N, wmma::mem_row_major);
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

    printf("\n=== WMMA cp.async Double Buffering HGEMM ===\n");
    printf("  M=N=K=%d\n", M);
    printf("  Block tile: BM=%d BN=%d BK=%d\n", BM, BN, BK);
    printf("  Warps: %dx%d=%d  (coarsening: %dx%d tiles/warp)\n",
           WARPS_M, WARPS_N, WARPS, COARSE_M, COARSE_N);
    printf("  Block: %d threads,  cp.async Double Buffer: ON\n", 32 * WARPS);

    half *hA = (half *)malloc(M * K * sizeof(half));
    half *hB = (half *)malloc(K * N * sizeof(half));
    for (int i = 0; i < M * K; i++) hA[i] = __float2half((rand() % 100) / 100.0f);
    for (int i = 0; i < K * N; i++) hB[i] = __float2half((rand() % 100) / 100.0f);

    half *dA, *dB;
    float *dC;
    CUDA_CHK(cudaMalloc(&dA, M * K * sizeof(half)));
    CUDA_CHK(cudaMalloc(&dB, K * N * sizeof(half)));
    CUDA_CHK(cudaMalloc(&dC, M * N * sizeof(float)));
    CUDA_CHK(cudaMemcpy(dA, hA, M * K * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHK(cudaMemcpy(dB, hB, K * N * sizeof(half), cudaMemcpyHostToDevice));

    dim3 block(32 * WARPS);
    dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN);
    printf("  Grid: (%d, %d)\n\n", grid.x, grid.y);

    cudaEvent_t start, stop;
    CUDA_CHK(cudaEventCreate(&start));
    CUDA_CHK(cudaEventCreate(&stop));

    hgemm_wmma_async<<<grid, block>>>(dA, dB, dC, M, N, K);
    CUDA_CHK(cudaDeviceSynchronize());

    CUDA_CHK(cudaEventRecord(start));
    hgemm_wmma_async<<<grid, block>>>(dA, dB, dC, M, N, K);
    CUDA_CHK(cudaGetLastError());
    CUDA_CHK(cudaEventRecord(stop));
    CUDA_CHK(cudaEventSynchronize(stop));

    float ms;
    CUDA_CHK(cudaEventElapsedTime(&ms, start, stop));
    double tflops = 2.0 * M * N * K / (ms / 1000.0) / 1e12;
    printf("  Time: %.4f ms  |  %.2f TFLOPS\n", ms, tflops);

    free(hA); free(hB);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return 0;
}
