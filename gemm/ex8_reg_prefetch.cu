#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

// ---- Kernel 10 风格参数 ----
#define BM  128
#define BN  128
#define BK  8

// ---- float4 辅助宏 ----
#define vload(v, addr)    v = *((float4*)(addr))
#define vstore(addr, v)   *((float4*)(addr)) = v
#define vfma4(c, a, b)    c.x += a.x*b; c.y += a.y*b; c.z += a.z*b; c.w += a.w*b;

// ---- 共享内存布局 (列主序, 128 行) ----
// sa: BK × BM = 8×128, 访问 sa[col * BM + row]
// sb: BK × BN = 8×128, 访问 sb[col * BN + n]
#define SA(row, col) sa[(col) * BM + (row)]
#define SB(n, k)     sb[(k) * BN + (n)]

// ============================================================
// Kernel 10: Register Prefetching (单 shared buffer, 无双缓冲)
//
//   线程映射: 1D 256 线程, warp-level tiling (4×2 warps)
//   B 转置存储 消除 bank conflict
//   两层 prefetch: Global→Reg (500 cyc) + Shared→Reg (20 cyc)
//   Shared memory: 8KB (sa:1KB + sb:1KB)
// ============================================================
__global__ __launch_bounds__(256)
void sgemm_k10(float* A, float* B, float* C, int M, int N, int K)
{
    int tx = threadIdx.x;
    int bx = blockIdx.x, by = blockIdx.y;

    // ---- Warp-level decomposition ----
    int warp_id = tx >> 5;          // 0..7
    int lane_id = tx & 31;          // 0..31
    int warp_row = warp_id & 3;     // 0..3
    int warp_col = warp_id >> 2;    // 0..1
    int row_w = lane_id & 3;        // 0..3
    int col_w = lane_id >> 2;       // 0..7

    // C 输出: 8×8 每线程
    int row_c = (warp_row << 5) + (row_w << 3);   // 0..127
    int col_c = (warp_col << 6) + (col_w << 3);   // 0..127

    // A 加载 (每线程 float4 × 1):
    //   128 rows × 2 float4/row = 256 float4 → 每线程 1 个
    int a_row = tx >> 1;            // 0..127 (哪一行)
    int a_col = (tx & 1) << 2;     // 0 或 4  (左/右半区间)

    // B 加载 (每线程 float4 × 1, 存为转置):
    //   沿 N 方向合并访问, 4 个连续 N 值 → 128 cols × 2 segments = 256
    int b_row = (tx & 1) << 2;     // 0 或 4  (K-dim 起点)
    int b_col = tx >> 1;           // 0..127 (N-dim 列号)

    // ---- 共享内存 (列主序, 仅 8KB) ----
    __shared__ float sa[BM * BK];   // 128×8 = 1024 floats
    __shared__ float sb[BN * BK];   // 128×8 = 1024 floats

    // ---- 寄存器 ----
    float4 Av1[2], Av2[2], Bv1[2], Bv2[2];
    float4 Cres[16];
    float4 pref_A, pref_B;

    memset(Cres, 0, sizeof(Cres));

    // ---- 各 tile 的 Global 基址 ----
    int K_tiles = K >> 3;  // K / BK
    float* A_base = A + by * BM * K;       // 本 block 的 A 行基址
    float* B_base = B + bx * BN;           // 本 block 的 B 列基址

    // ============================================================
    // Prologue: 加载 tile 0 → shared, 预取 inner_k=0 → 寄存器
    // ============================================================
    {
        float* ptr_A = A_base;                // t = 0
        float* ptr_B = B_base + 0;            // t = 0, row=0
        vload(pref_A, &ptr_A[a_row * K + a_col]);
        // B 行主序下跨 K 行不连续, 必须逐行读
        int bb = b_row * N + b_col;
        pref_B.x = ptr_B[bb];
        pref_B.y = ptr_B[bb + N];
        pref_B.z = ptr_B[bb + 2*N];
        pref_B.w = ptr_B[bb + 3*N];

        // 写入 shared (A 逐元素散列到列主序, B 转置存)
        SA(a_row, a_col + 0) = pref_A.x;
        SA(a_row, a_col + 1) = pref_A.y;
        SA(a_row, a_col + 2) = pref_A.z;
        SA(a_row, a_col + 3) = pref_A.w;
        SB(b_col, b_row + 0) = pref_B.x;
        SB(b_col, b_row + 1) = pref_B.y;
        SB(b_col, b_row + 2) = pref_B.z;
        SB(b_col, b_row + 3) = pref_B.w;
    }
    __syncthreads();

    // 从 shared 预取 inner_k=0 到寄存器
    vload(Av1[0], &SA(row_c, 0));
    vload(Av2[0], &SA(row_c + 4, 0));
    vload(Bv1[0], &SB(col_c, 0));
    vload(Bv2[0], &SB(col_c + 4, 0));

    // ============================================================
    // Main Loop
    // ============================================================
    for (int kc = 0; kc < K_tiles; kc++) {
        // ---- ① LDG: 预取下一个 tile → 寄存器 (与计算并行) ----
        int inc = (kc + 1) % K_tiles;
        if (kc + 1 < K_tiles) {
            float* next_A = A_base + inc * BK;
            float* next_B = B_base + inc * BK * N;
            vload(pref_A, &next_A[a_row * K + a_col]);
            int bb = b_row * N + b_col;
            pref_B.x = next_B[bb];
            pref_B.y = next_B[bb + N];
            pref_B.z = next_B[bb + 2*N];
            pref_B.w = next_B[bb + 3*N];
        }

        // ---- ② 内层循环: 寄存器乒乓 (shared→reg 预取) ----
        #pragma unroll
        for (int ik = 0; ik < BK; ik++) {
            int next_ik = (ik + 1) & 7;

            // 预取下一个 inner_k (shared → 寄存器)
            vload(Av1[(ik + 1) & 1], &SA(row_c, next_ik));
            vload(Av2[(ik + 1) & 1], &SA(row_c + 4, next_ik));
            vload(Bv1[(ik + 1) & 1], &SB(col_c, next_ik));
            vload(Bv2[(ik + 1) & 1], &SB(col_c + 4, next_ik));

            // 计算当前 (16 个 float4 FMA = 8×8 = 64 FMA)
            float4 a1 = Av1[ik & 1], a2 = Av2[ik & 1];
            float4 b1 = Bv1[ik & 1], b2 = Bv2[ik & 1];

            vfma4(Cres[0],  a1, b1.x);  vfma4(Cres[1],  a2, b1.x);
            vfma4(Cres[2],  a1, b1.y);  vfma4(Cres[3],  a2, b1.y);
            vfma4(Cres[4],  a1, b1.z);  vfma4(Cres[5],  a2, b1.z);
            vfma4(Cres[6],  a1, b1.w);  vfma4(Cres[7],  a2, b1.w);
            vfma4(Cres[8],  a1, b2.x);  vfma4(Cres[9],  a2, b2.x);
            vfma4(Cres[10], a1, b2.y);  vfma4(Cres[11], a2, b2.y);
            vfma4(Cres[12], a1, b2.z);  vfma4(Cres[13], a2, b2.z);
            vfma4(Cres[14], a1, b2.w);  vfma4(Cres[15], a2, b2.w);
        }

        // ---- ③ 写入下一 tile 到 shared ----
        if (kc + 1 < K_tiles) {
            __syncthreads();
            SA(a_row, a_col + 0) = pref_A.x;
            SA(a_row, a_col + 1) = pref_A.y;
            SA(a_row, a_col + 2) = pref_A.z;
            SA(a_row, a_col + 3) = pref_A.w;
            SB(b_col, b_row + 0) = pref_B.x;
            SB(b_col, b_row + 1) = pref_B.y;
            SB(b_col, b_row + 2) = pref_B.z;
            SB(b_col, b_row + 3) = pref_B.w;
            __syncthreads();
        }

        // ---- ④ 预取新 tile 的 inner_k=0 ----
        vload(Av1[0], &SA(row_c, 0));
        vload(Av2[0], &SA(row_c + 4, 0));
        vload(Bv1[0], &SB(col_c, 0));
        vload(Bv2[0], &SB(col_c + 4, 0));
    }

    // ============================================================
    // Write Back: 8×8 = 64 outputs
    //   注意: row-major 下 C[i*N+j], 同一行连续列才合并访问
    //   Cres 存的是同列不同行(不连续), 所以用单独 float 写入
    // ============================================================
    float* C_block = C + by * BM * N + bx * BN;

    #define W(x, ro, co) C_block[(row_c + ro) * N + col_c + co] = x

    // col_c+0
    W(Cres[0].x, 0,0); W(Cres[0].y, 1,0); W(Cres[0].z, 2,0); W(Cres[0].w, 3,0);
    W(Cres[1].x, 4,0); W(Cres[1].y, 5,0); W(Cres[1].z, 6,0); W(Cres[1].w, 7,0);
    // col_c+1
    W(Cres[2].x, 0,1); W(Cres[2].y, 1,1); W(Cres[2].z, 2,1); W(Cres[2].w, 3,1);
    W(Cres[3].x, 4,1); W(Cres[3].y, 5,1); W(Cres[3].z, 6,1); W(Cres[3].w, 7,1);
    // col_c+2
    W(Cres[4].x, 0,2); W(Cres[4].y, 1,2); W(Cres[4].z, 2,2); W(Cres[4].w, 3,2);
    W(Cres[5].x, 4,2); W(Cres[5].y, 5,2); W(Cres[5].z, 6,2); W(Cres[5].w, 7,2);
    // col_c+3
    W(Cres[6].x, 0,3); W(Cres[6].y, 1,3); W(Cres[6].z, 2,3); W(Cres[6].w, 3,3);
    W(Cres[7].x, 4,3); W(Cres[7].y, 5,3); W(Cres[7].z, 6,3); W(Cres[7].w, 7,3);
    // col_c+4
    W(Cres[8].x, 0,4); W(Cres[8].y, 1,4); W(Cres[8].z, 2,4); W(Cres[8].w, 3,4);
    W(Cres[9].x, 4,4); W(Cres[9].y, 5,4); W(Cres[9].z, 6,4); W(Cres[9].w, 7,4);
    // col_c+5
    W(Cres[10].x, 0,5); W(Cres[10].y, 1,5); W(Cres[10].z, 2,5); W(Cres[10].w, 3,5);
    W(Cres[11].x, 4,5); W(Cres[11].y, 5,5); W(Cres[11].z, 6,5); W(Cres[11].w, 7,5);
    // col_c+6
    W(Cres[12].x, 0,6); W(Cres[12].y, 1,6); W(Cres[12].z, 2,6); W(Cres[12].w, 3,6);
    W(Cres[13].x, 4,6); W(Cres[13].y, 5,6); W(Cres[13].z, 6,6); W(Cres[13].w, 7,6);
    // col_c+7
    W(Cres[14].x, 0,7); W(Cres[14].y, 1,7); W(Cres[14].z, 2,7); W(Cres[14].w, 3,7);
    W(Cres[15].x, 4,7); W(Cres[15].y, 5,7); W(Cres[15].z, 6,7); W(Cres[15].w, 7,7);

    #undef W
}

// ============================================================
// 计时 + 正确性验证
// ============================================================
#define CUDA_CHK(call) do {                                        \
    cudaError_t _e_ = (call);                                    \
    if (_e_ != cudaSuccess) {                                    \
        printf("CUDA ERROR: %s\n", cudaGetErrorString(_e_));   \
        exit(1);                                                 \
    }                                                            \
} while(0)

float timeKernel(void (*fn)(float*,float*,float*,int,int,int),
                 dim3 g, dim3 b, float* dA, float* dB, float* dC,
                 int M, int N, int K, const char* label)
{
    cudaEvent_t s, e;
    cudaEventCreate(&s); cudaEventCreate(&e);
    fn<<<g,b>>>(dA,dB,dC,M,N,K); CUDA_CHK(cudaDeviceSynchronize());
    cudaEventRecord(s);
    fn<<<g,b>>>(dA,dB,dC,M,N,K); CUDA_CHK(cudaGetLastError());
    cudaEventRecord(e);
    CUDA_CHK(cudaEventSynchronize(e));
    float ms; cudaEventElapsedTime(&ms,s,e);
    printf("  %-30s  %.4f ms\n", label, ms);
    cudaEventDestroy(s); cudaEventDestroy(e);
    return ms;
}

// ---- CPU 参考 SGEMM (行主序) ----
void sgemm_cpu(float* A, float* B, float* C, int M, int N, int K) {
    memset(C, 0, M * N * sizeof(float));
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++)
            for (int k = 0; k < K; k++)
                C[i * N + j] += A[i * K + k] * B[k * N + j];
}

void init(float* m, int n) { for(int i=0;i<n;i++) m[i]=(rand()%100)/100.0f; }

int main()
{
    srand(42);
    int M = 4096, N = 4096, K = 4096;
    size_t szA=M*K*sizeof(float), szB=K*N*sizeof(float), szC=M*N*sizeof(float);

    // GPU 属性
    int dev; cudaGetDevice(&dev);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);
    printf("\n=== Kernel 10: Register Prefetching ===\n");
    printf("  GPU: %s, SM=%d\n", prop.name, prop.multiProcessorCount);
    printf("  BM=%d BN=%d BK=%d  Shared=8KB  Block=256  Grid=%dx%d\n",
           BM, BN, BK, (N+BN-1)/BN, (M+BM-1)/BM);

    float *hA=(float*)malloc(szA), *hB=(float*)malloc(szB);
    float *hC_cpu=(float*)malloc(szC), *hC_gpu=(float*)malloc(szC);
    init(hA, M*K); init(hB, K*N);

    float *dA, *dB, *dC;
    CUDA_CHK(cudaMalloc(&dA, szA));
    CUDA_CHK(cudaMalloc(&dB, szB));
    CUDA_CHK(cudaMalloc(&dC, szC));
    CUDA_CHK(cudaMemcpy(dA, hA, szA, cudaMemcpyHostToDevice));
    CUDA_CHK(cudaMemcpy(dB, hB, szB, cudaMemcpyHostToDevice));

    dim3 block(256);
    dim3 grid((N+BN-1)/BN, (M+BM-1)/BM);

    float ms = timeKernel(sgemm_k10, grid, block, dA,dB,dC, M,N,K,
                          "Kernel 10: reg prefetch");

    // 正确性验证
    CUDA_CHK(cudaMemcpy(hC_gpu, dC, szC, cudaMemcpyDeviceToHost));
    sgemm_cpu(hA, hB, hC_cpu, M, N, K);

    float maxDiff = 0, sumCPU = 0;
    for (int i = 0; i < M*N; i++) {
        float d = fabsf(hC_gpu[i] - hC_cpu[i]);
        if (d > maxDiff) maxDiff = d;
        sumCPU += fabsf(hC_cpu[i]);
    }
    float tol = sumCPU / (M*N) * 1e-3f;
    printf("  Max diff: %e (tol=%.2e) → %s\n",
           maxDiff, tol, maxDiff < tol ? "PASS" : "FAIL");

    double gflops = 2.0 * M * N * K / (ms * 1e6);
    printf("  Performance: %.1f GFLOPS\n", gflops);

    CUDA_CHK(cudaFree(dA)); CUDA_CHK(cudaFree(dB)); CUDA_CHK(cudaFree(dC));
    free(hA); free(hB); free(hC_cpu); free(hC_gpu);
    return 0;
}
