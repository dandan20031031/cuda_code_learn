#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

#define BM 128
#define BN 128
#define BK 32

#define BK_PAD (BK + 8)
#define BN_PAD (BN + 8)

#define WARPS_M 2
#define WARPS_N 4
#define WARPS   (WARPS_M * WARPS_N)

#define COARSE_M (BM / WMMA_M / WARPS_M)
#define COARSE_N (BN / WMMA_N / WARPS_N)

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

__device__ __forceinline__ void load_As_async(half As[BM][BK_PAD], const half *A,
                                              int K, int m_off, int kb)
{
    uint32_t smem = (uint32_t)__cvta_generic_to_shared(As);
    constexpr int CHUNKS_PER_ROW = BK / 8;
    constexpr int TOTAL_CHUNKS   = BM * CHUNKS_PER_ROW;
    #pragma unroll
    for (int c = threadIdx.x; c < TOTAL_CHUNKS; c += blockDim.x) {
        int row  = c / CHUNKS_PER_ROW;
        int col8 = c % CHUNKS_PER_ROW;
        uint32_t dst = smem + (row * BK_PAD + col8 * 8) * sizeof(half);
        const half *src = &A[(m_off + row) * K + kb + col8 * 8];
        cp_async16(dst, src);
    }
}

__device__ __forceinline__ void load_Bs_async(half Bs[BK][BN_PAD], const half *B,
                                              int N, int n_off, int kb)
{
    uint32_t smem = (uint32_t)__cvta_generic_to_shared(Bs);
    constexpr int CHUNKS_PER_ROW = BN / 8;
    constexpr int TOTAL_CHUNKS   = BK * CHUNKS_PER_ROW;
    #pragma unroll
    for (int c = threadIdx.x; c < TOTAL_CHUNKS; c += blockDim.x) {
        int row  = c / CHUNKS_PER_ROW;
        int col8 = c % CHUNKS_PER_ROW;
        uint32_t dst = smem + (row * BN_PAD + col8 * 8) * sizeof(half);
        const half *src = &B[(kb + row) * N + n_off + col8 * 8];
        cp_async16(dst, src);
    }
}

__global__ void hgemm_wmma_tile128_halfacc(
    const half *A,
    const half *B,
    half        *C,
    int M, int N, int K)
{
    int warp_id = threadIdx.x / 32;
    int warp_m  = warp_id / WARPS_N;
    int warp_n  = warp_id % WARPS_N;

    int c_row_base = blockIdx.x * BM + warp_m * COARSE_M * WMMA_M;
    int c_col_base = blockIdx.y * BN + warp_n * COARSE_N * WMMA_N;
    if (c_row_base >= M || c_col_base >= N) return;

    __shared__ half As[2][BM][BK_PAD];
    __shared__ half Bs[2][BK][BN_PAD];

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> c_frag[COARSE_M][COARSE_N];
    #pragma unroll
    for (int mi = 0; mi < COARSE_M; mi++)
        #pragma unroll
        for (int ni = 0; ni < COARSE_N; ni++)
            wmma::fill_fragment(c_frag[mi][ni], 0.0f);

    load_As_async(As[0], A, K, blockIdx.x * BM, 0);
    load_Bs_async(Bs[0], B, N, blockIdx.y * BN, 0);
    cp_async_commit();

    const int n_iters = K / BK;
    for (int it = 0; it < n_iters; it++) {
        int cur = it & 1;
        int nxt = cur ^ 1;

        if (it + 1 < n_iters) {
            int next_kb = (it + 1) * BK;
            load_As_async(As[nxt], A, K, blockIdx.x * BM, next_kb);
            load_Bs_async(Bs[nxt], B, N, blockIdx.y * BN, next_kb);
            cp_async_commit();
        }

        cp_async_wait<1>();
        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < BK; kk += WMMA_K) {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag[COARSE_M];
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag[COARSE_N];

            #pragma unroll
            for (int mi = 0; mi < COARSE_M; mi++) {
                int a_off = (warp_m * COARSE_M + mi) * WMMA_M;
                wmma::load_matrix_sync(a_frag[mi], &As[cur][a_off][kk], BK_PAD);
            }
            #pragma unroll
            for (int ni = 0; ni < COARSE_N; ni++) {
                int b_off = (warp_n * COARSE_N + ni) * WMMA_N;
                wmma::load_matrix_sync(b_frag[ni], &Bs[cur][kk][b_off], BN_PAD);
            }

            #pragma unroll
            for (int mi = 0; mi < COARSE_M; mi++)
                #pragma unroll
                for (int ni = 0; ni < COARSE_N; ni++)
                    wmma::mma_sync(c_frag[mi][ni], a_frag[mi], b_frag[ni], c_frag[mi][ni]);
        }
        __syncthreads();
    }

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

int main() {
    // 用较小 M=N 但保留完整 K，CPU 能算动，又能反映 K 长累加链的精度损失
    const int M = 512, N = 512, K = 4096;

    printf("\n=== v8 half-acc 正确性校验 (M=N=512, K=4096) ===\n");

    half *hA = (half *)malloc(M * K * sizeof(half));
    half *hB = (half *)malloc(K * N * sizeof(half));
    for (int i = 0; i < M * K; i++) hA[i] = __float2half((rand() % 100) / 100.0f);
    for (int i = 0; i < K * N; i++) hB[i] = __float2half((rand() % 100) / 100.0f);

    half *dA, *dB, *dC;
    CUDA_CHK(cudaMalloc(&dA, M * K * sizeof(half)));
    CUDA_CHK(cudaMalloc(&dB, K * N * sizeof(half)));
    CUDA_CHK(cudaMalloc(&dC, M * N * sizeof(half)));
    CUDA_CHK(cudaMemcpy(dA, hA, M * K * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHK(cudaMemcpy(dB, hB, K * N * sizeof(half), cudaMemcpyHostToDevice));

    dim3 block(32 * WARPS);
    dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN);
    hgemm_wmma_tile128_halfacc<<<grid, block>>>(dA, dB, dC, M, N, K);
    CUDA_CHK(cudaDeviceSynchronize());
    CUDA_CHK(cudaGetLastError());

    half *hC = (half *)malloc(M * N * sizeof(half));
    CUDA_CHK(cudaMemcpy(hC, dC, M * N * sizeof(half), cudaMemcpyDeviceToHost));

    // CPU 参考：half 输入 + double 精确累加（ground truth）
    double *C_ref = (double *)malloc(M * N * sizeof(double));
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            double s = 0.0;
            for (int k = 0; k < K; k++)
                s += (double)__half2float(hA[i * K + k]) * (double)__half2float(hB[k * N + j]);
            C_ref[i * N + j] = s;
        }
    }

    double max_diff = 0.0, sum_diff = 0.0, sum_ref = 0.0;
    for (int i = 0; i < M * N; i++) {
        double g = (double)__half2float(hC[i]);
        double d = fabs(g - C_ref[i]);
        if (d > max_diff) max_diff = d;
        sum_diff += d;
        sum_ref += fabs(C_ref[i]);
    }
    double avg_diff = sum_diff / (M * N);
    double avg_ref  = sum_ref / (M * N);
    printf("  Max diff  = %f\n", max_diff);
    printf("  Avg diff  = %f\n", avg_diff);
    printf("  Avg |ref| = %f\n", avg_ref);
    printf("  Avg relative error = %.4f%%\n", 100.0 * avg_diff / avg_ref);

    free(hA); free(hB); free(hC); free(C_ref);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return 0;
}
