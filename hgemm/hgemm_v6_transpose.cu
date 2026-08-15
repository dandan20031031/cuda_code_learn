#include <stdio.h>
#include <stdlib.h>
#include <math.h>
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

// v6 关键：A 和 B 统一用 BK_PAD 作为行 stride。
// B 转置存储 [N][K]（col_major）
// 从源头消除 v5 里 B 的竖读 bank conflict（v5 的 BN_PAD 被去掉）。
#define BK_PAD (BK + 8)   // 40 halfs = 80 字节，打破 bank 相位对齐

#define WARPS_M 2
#define WARPS_N 2
#define WARPS   (WARPS_M * WARPS_N)      // 4 warps

#define COARSE_M (BM / WMMA_M / WARPS_M) // 2
#define COARSE_N (BN / WMMA_N / WARPS_N) // 2

// ---- cp.async 内联 PTX 封装 ----
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

// ---- 异步搬 A tile: HBM → shared，A 保持 row_major [M][K] ----
__device__ __forceinline__ void load_As_async(half As[BM][BK_PAD], const half *A,
                                              int K, int m_off, int kb)
{
    uint32_t smem = (uint32_t)__cvta_generic_to_shared(As);
    constexpr int CHUNKS_PER_ROW = BK / 8;        // 每行 32 half = 4 个 16 字节块
    constexpr int TOTAL_CHUNKS   = BM * CHUNKS_PER_ROW;
    #pragma unroll
    for (int c = threadIdx.x; c < TOTAL_CHUNKS; c += blockDim.x) {
        int row  = c / CHUNKS_PER_ROW;            // M 方向
        int col8 = c % CHUNKS_PER_ROW;            // K 方向（8 half 为单位）
        uint32_t dst = smem + (row * BK_PAD + col8 * 8) * sizeof(half);
        const half *src = &A[(m_off + row) * K + kb + col8 * 8];
        cp_async16(dst, src);
    }
}

// ---- 异步搬 B tile：B 转置存储 [N][K]（col_major）----
// global B 的物理布局是 B[n*K + k] = 逻辑 B[k][n]
// 这样 shared Bs 存 [N][K]，col_major 读是"行内连续"（顺读）
__device__ __forceinline__ void load_Bs_async(half Bs[BN][BK_PAD], const half *B,
                                              int K, int n_off, int kb)
{
    uint32_t smem = (uint32_t)__cvta_generic_to_shared(Bs);
    constexpr int CHUNKS_PER_ROW = BK / 8;        // K 方向每行 32 half = 4 个 16 字节块
    constexpr int TOTAL_CHUNKS   = BN * CHUNKS_PER_ROW;
    #pragma unroll
    for (int c = threadIdx.x; c < TOTAL_CHUNKS; c += blockDim.x) {
        int row  = c / CHUNKS_PER_ROW;            // N 方向
        int col8 = c % CHUNKS_PER_ROW;            // K 方向（8 half 为单位）
        uint32_t dst = smem + (row * BK_PAD + col8 * 8) * sizeof(half);
        const half *src = &B[(n_off + row) * K + kb + col8 * 8];  // B 是 [N][K]
        cp_async16(dst, src);
    }
}

__global__ void hgemm_wmma_transpose(
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

    // 双 buffer：A 是 [M][K]，B 转置成 [N][K]，两者共用 BK_PAD stride
    __shared__ half As[2][BM][BK_PAD];
    __shared__ half Bs[2][BN][BK_PAD];

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag[COARSE_M][COARSE_N];
    #pragma unroll
    for (int mi = 0; mi < COARSE_M; mi++)
        #pragma unroll
        for (int ni = 0; ni < COARSE_N; ni++)
            wmma::fill_fragment(c_frag[mi][ni], 0.0f);

    // 预取第 0 个 K tile 到 buffer[0]
    load_As_async(As[0], A, K, blockIdx.x * BM, 0);
    load_Bs_async(Bs[0], B, K, blockIdx.y * BN, 0);
    cp_async_commit();

    const int n_iters = K / BK;
    for (int it = 0; it < n_iters; it++) {
        int cur = it & 1;
        int nxt = cur ^ 1;

        // 预取下一个 K tile（异步）
        if (it + 1 < n_iters) {
            int next_kb = (it + 1) * BK;
            load_As_async(As[nxt], A, K, blockIdx.x * BM, next_kb);
            load_Bs_async(Bs[nxt], B, K, blockIdx.y * BN, next_kb);
            cp_async_commit();
        }

        // 等当前 tile 到位
        cp_async_wait<1>();
        __syncthreads();

        // ---- 计算 ----
        #pragma unroll
        for (int mi = 0; mi < COARSE_M; mi++) {
            #pragma unroll
            for (int ni = 0; ni < COARSE_N; ni++) {
                int a_off = (warp_m * COARSE_M + mi) * WMMA_M;
                int b_off = (warp_n * COARSE_N + ni) * WMMA_N;
                #pragma unroll
                for (int kk = 0; kk < BK; kk += WMMA_K) {
                    wmma::load_matrix_sync(a_frag, &As[cur][a_off][kk], BK_PAD);
                    // B 转置后：Bs[N][K]，col_major 读，ldm = BK_PAD
                    wmma::load_matrix_sync(b_frag, &Bs[cur][b_off][kk], BK_PAD);
                    wmma::mma_sync(c_frag[mi][ni], a_frag, b_frag, c_frag[mi][ni]);
                }
            }
        }
        __syncthreads();
    }

    // 写回 global memory
    #pragma unroll
    for (int mi = 0; mi < COARSE_M; mi++) {
        #pragma unroll
        for (int ni = 0; ni < COARSE_N; ni++) {
            int cr = c_row_base + mi * WMMA_M;
            int cc = c_col_base + ni * WMMA_N;
            if (cr < M && cc < N)
                wmma::store_matrix_sync(C + cr * N + cc, c_frag[mi][ni], N, wmma::mem_row_major);
        }
    }
}

#define CUDA_CHK(call) do { \
    cudaError_t _e_ = (call); \
    if (_e_ != cudaSuccess) { \
        printf("CUDA ERROR at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e_)); \
        exit(1); \
    } \
} while(0)

// CPU 参考：C[m][n] = sum_k A[m][k] * B[k][n]
// A 是 [M][K] row-major，B 是 [N][K] col-major（转置存储）
void cpu_hgemm(const half *A, const half *B, float *C, int M, int N, int K) {
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++)
                sum += __half2float(A[m * K + k]) * __half2float(B[n * K + k]);
            C[m * N + n] = sum;
        }
}

// 用给定 M/N/K 跑一次 kernel 并返回 TFLOPS
double run_kernel(int M, int N, int K, half *dA, half *dB, float *dC) {
    dim3 block(32 * WARPS);
    dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN);

    cudaEvent_t start, stop;
    CUDA_CHK(cudaEventCreate(&start));
    CUDA_CHK(cudaEventCreate(&stop));

    hgemm_wmma_transpose<<<grid, block>>>(dA, dB, dC, M, N, K);
    CUDA_CHK(cudaDeviceSynchronize());

    CUDA_CHK(cudaEventRecord(start));
    hgemm_wmma_transpose<<<grid, block>>>(dA, dB, dC, M, N, K);
    CUDA_CHK(cudaGetLastError());
    CUDA_CHK(cudaEventRecord(stop));
    CUDA_CHK(cudaEventSynchronize(stop));

    float ms;
    CUDA_CHK(cudaEventElapsedTime(&ms, start, stop));
    double tflops = 2.0 * M * N * K / (ms / 1000.0) / 1e12;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return tflops;
}

int main() {
    printf("\n=== WMMA cp.async + B 转置 (v6) ===\n");
    printf("  Block tile: BM=%d BN=%d BK=%d  (A/B 统一 stride BK_PAD=%d)\n", BM, BN, BK, BK_PAD);
    printf("  Warps: %dx%d=%d  (coarsening: %dx%d tiles/warp)\n",
           WARPS_M, WARPS_N, WARPS, COARSE_M, COARSE_N);

    // ============ 1. 正确性校验（小矩阵）============
    {
        const int M = 512, N = 512, K = 512;
        printf("\n[校验] M=N=K=%d\n", M);

        half *hA = (half *)malloc(M * K * sizeof(half));
        half *hB = (half *)malloc(N * K * sizeof(half));  // 注意 B 是 [N][K]
        float *hC = (float *)malloc(M * N * sizeof(float));
        float *hRef = (float *)malloc(M * N * sizeof(float));

        for (int i = 0; i < M * K; i++) hA[i] = __float2half((rand() % 100) / 100.0f);
        for (int i = 0; i < N * K; i++) hB[i] = __float2half((rand() % 100) / 100.0f);

        half *dA, *dB;
        float *dC;
        CUDA_CHK(cudaMalloc(&dA, M * K * sizeof(half)));
        CUDA_CHK(cudaMalloc(&dB, N * K * sizeof(half)));
        CUDA_CHK(cudaMalloc(&dC, M * N * sizeof(float)));
        CUDA_CHK(cudaMemcpy(dA, hA, M * K * sizeof(half), cudaMemcpyHostToDevice));
        CUDA_CHK(cudaMemcpy(dB, hB, N * K * sizeof(half), cudaMemcpyHostToDevice));

        dim3 block(32 * WARPS);
        dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN);
        hgemm_wmma_transpose<<<grid, block>>>(dA, dB, dC, M, N, K);
        CUDA_CHK(cudaDeviceSynchronize());

        CUDA_CHK(cudaMemcpy(hC, dC, M * N * sizeof(float), cudaMemcpyDeviceToHost));
        cpu_hgemm(hA, hB, hRef, M, N, K);

        double max_diff = 0.0;
        for (int i = 0; i < M * N; i++)
            max_diff = fmax(max_diff, fabs(hC[i] - hRef[i]));
        printf("  max diff = %.6f  %s\n", max_diff, max_diff < 1.0 ? "(正确)" : "(错误!)");

        free(hA); free(hB); free(hC); free(hRef);
        cudaFree(dA); cudaFree(dB); cudaFree(dC);
    }

    // ============ 2. 性能测试（大矩阵）============
    {
        const int M = 4096, N = 4096, K = 4096;
        printf("\n[性能] M=N=K=%d\n", M);

        half *dA, *dB;
        float *dC;
        CUDA_CHK(cudaMalloc(&dA, M * K * sizeof(half)));
        CUDA_CHK(cudaMalloc(&dB, N * K * sizeof(half)));
        CUDA_CHK(cudaMalloc(&dC, M * N * sizeof(float)));

        // 随机初始化（GPU 端，避免 host 大数组）
        half *hA = (half *)malloc(M * K * sizeof(half));
        half *hB = (half *)malloc(N * K * sizeof(half));
        for (int i = 0; i < M * K; i++) hA[i] = __float2half((rand() % 100) / 100.0f);
        for (int i = 0; i < N * K; i++) hB[i] = __float2half((rand() % 100) / 100.0f);
        CUDA_CHK(cudaMemcpy(dA, hA, M * K * sizeof(half), cudaMemcpyHostToDevice));
        CUDA_CHK(cudaMemcpy(dB, hB, N * K * sizeof(half), cudaMemcpyHostToDevice));
        free(hA); free(hB);

        double tflops = run_kernel(M, N, K, dA, dB, dC);
        printf("  %.2f TFLOPS  (v5 为 89.99)\n", tflops);

        cudaFree(dA); cudaFree(dB); cudaFree(dC);
    }

    return 0;
}
