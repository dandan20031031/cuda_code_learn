#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define TILE 16
#define STRIDE 4
#define TILE_K (TILE * STRIDE)
#define RK 4

// ============================================================
// STRIDE=4 + float4 (from ex6)
// ============================================================
__global__ void sgemm_stride4(float* A, float* B, float* C, int M, int N, int K)
{
    int tx = threadIdx.x, ty = threadIdx.y;
    int row0 = blockIdx.y * TILE_K + ty * STRIDE;
    int col0 = blockIdx.x * TILE_K + tx * STRIDE;

    __shared__ float As[TILE_K][TILE_K];
    __shared__ float Bs[TILE_K][TILE_K];
    float reg[STRIDE][STRIDE] = {0.0f};

    for (int t = 0; t < K; t += TILE_K) {
        for (int di = 0; di < STRIDE; di++) {
            int r = row0 + di;
            int sm_row = ty * STRIDE + di;
            int sm_col = tx * STRIDE;
            if (r < M && t + sm_col + STRIDE - 1 < K) {
                float4 tmp = *(float4*)&A[r * K + t + sm_col];
                As[sm_row][sm_col+0]=tmp.x; As[sm_row][sm_col+1]=tmp.y;
                As[sm_row][sm_col+2]=tmp.z; As[sm_row][sm_col+3]=tmp.w;
            } else {
                for (int dj = 0; dj < STRIDE; dj++)
                    As[sm_row][sm_col+dj] =
                        (r < M && t + sm_col + dj < K) ? A[r * K + t + sm_col + dj] : 0.0f;
            }
        }
        for (int di = 0; di < STRIDE; di++) {
            int sm_row = ty * STRIDE + di, sm_col = tx * STRIDE;
            int kr = t + sm_row;
            if (col0 + STRIDE - 1 < N && kr < K) {
                float4 tmp = *(float4*)&B[kr * N + col0];
                Bs[sm_row][sm_col+0]=tmp.x; Bs[sm_row][sm_col+1]=tmp.y;
                Bs[sm_row][sm_col+2]=tmp.z; Bs[sm_row][sm_col+3]=tmp.w;
            } else {
                for (int dj = 0; dj < STRIDE; dj++)
                    Bs[sm_row][sm_col+dj] =
                        (kr < K && col0 + dj < N) ? B[kr * N + col0 + dj] : 0.0f;
            }
        }
        __syncthreads();

        for (int di = 0; di < STRIDE; di++)
            for (int dj = 0; dj < STRIDE; dj++)
                for (int k = 0; k < TILE_K; k++)
                    reg[di][dj] += As[ty*STRIDE+di][k] * Bs[k][tx*STRIDE+dj];

        __syncthreads();
    }
    for (int di = 0; di < STRIDE; di++)
        for (int dj = 0; dj < STRIDE; dj++) {
            int r = row0+di, c = col0+dj;
            if (r < M && c < N) C[r*N + c] = reg[di][dj];
        }
}

// ============================================================
// STRIDE=4 + Register Tiling (RK=4)
//   K tile 内再分 4x4 register block，先搬到 reg 再算
//   减少 shared memory 读，提升算术强度
// ============================================================
__global__ void sgemm_stride4_regtile(float* A, float* B, float* C, int M, int N, int K)
{
    int tx = threadIdx.x, ty = threadIdx.y;
    int row0 = blockIdx.y * TILE_K + ty * STRIDE;
    int col0 = blockIdx.x * TILE_K + tx * STRIDE;

    __shared__ float As[TILE_K][TILE_K];
    __shared__ float Bs[TILE_K][TILE_K];
    float reg[STRIDE][STRIDE] = {0.0f};
    float reg_A[STRIDE][RK];          // 4x4 = 16 reg
    float reg_B[RK][STRIDE];          // 4x4 = 16 reg

    for (int t = 0; t < K; t += TILE_K) {
        for (int di = 0; di < STRIDE; di++) {
            int r = row0 + di;
            int sm_row = ty * STRIDE + di;
            int sm_col = tx * STRIDE;
            if (r < M && t + sm_col + STRIDE - 1 < K) {
                float4 tmp = *(float4*)&A[r * K + t + sm_col];
                As[sm_row][sm_col+0]=tmp.x; As[sm_row][sm_col+1]=tmp.y;
                As[sm_row][sm_col+2]=tmp.z; As[sm_row][sm_col+3]=tmp.w;
            } else {
                for (int dj = 0; dj < STRIDE; dj++)
                    As[sm_row][sm_col+dj] =
                        (r < M && t + sm_col + dj < K) ? A[r * K + t + sm_col + dj] : 0.0f;
            }
        }
        for (int di = 0; di < STRIDE; di++) {
            int sm_row = ty * STRIDE + di, sm_col = tx * STRIDE;
            int kr = t + sm_row;
            if (col0 + STRIDE - 1 < N && kr < K) {
                float4 tmp = *(float4*)&B[kr * N + col0];
                Bs[sm_row][sm_col+0]=tmp.x; Bs[sm_row][sm_col+1]=tmp.y;
                Bs[sm_row][sm_col+2]=tmp.z; Bs[sm_row][sm_col+3]=tmp.w;
            } else {
                for (int dj = 0; dj < STRIDE; dj++)
                    Bs[sm_row][sm_col+dj] =
                        (kr < K && col0 + dj < N) ? B[kr * N + col0 + dj] : 0.0f;
            }
        }
        __syncthreads();

        // ---- Register Tiling: TILE_K=64 / RK=4 = 16 sub-tiles ----
        for (int kk = 0; kk < TILE_K; kk += RK) {

            for (int di = 0; di < STRIDE; di++)
                for (int k = 0; k < RK; k++)
                    reg_A[di][k] = As[ty * STRIDE + di][kk + k];

            for (int k = 0; k < RK; k++)
                for (int dj = 0; dj < STRIDE; dj++)
                    reg_B[k][dj] = Bs[kk + k][tx * STRIDE + dj];

            for (int di = 0; di < STRIDE; di++)
                for (int dj = 0; dj < STRIDE; dj++)
                    for (int k = 0; k < RK; k++)
                        reg[di][dj] += reg_A[di][k] * reg_B[k][dj];
        }

        __syncthreads();
    }
    for (int di = 0; di < STRIDE; di++)
        for (int dj = 0; dj < STRIDE; dj++) {
            int r = row0+di, c = col0+dj;
            if (r < M && c < N) C[r*N + c] = reg[di][dj];
        }
}

// ============================================================
typedef void (*kernel_fn)(float*, float*, float*, int, int, int);
float timeKernel(kernel_fn fn, dim3 g, dim3 b, float* dA, float* dB, float* dC,
                 int M, int N, int K, const char* label)
{
    cudaEvent_t s, e;
    cudaEventCreate(&s); cudaEventCreate(&e);
    fn<<<g,b>>>(dA,dB,dC,M,N,K); cudaDeviceSynchronize();
    cudaEventRecord(s);
    fn<<<g,b>>>(dA,dB,dC,M,N,K);
    cudaEventRecord(e);
    cudaEventSynchronize(e);
    float ms; cudaEventElapsedTime(&ms,s,e);
    printf("  %-20s  %.4f ms\n", label, ms);
    cudaEventDestroy(s); cudaEventDestroy(e);
    return ms;
}

void init(float* m, int n) { for(int i=0;i<n;i++) m[i]=(rand()%100)/100.0f; }

int main()
{
    srand(42);
    int M = 4096, N = 4096, K = 4096;

    size_t szA=M*K*sizeof(float), szB=K*N*sizeof(float), szC=M*N*sizeof(float);
    printf("\n=== SGEMM STRIDE=4 vs STRIDE=4 + Register Tiling (RK=%d): %dx%dx%d ===\n",RK,M,N,K);

    float *hA=(float*)malloc(szA), *hB=(float*)malloc(szB);
    init(hA, M*K); init(hB, K*N);

    float *dA, *dB, *dC1, *dC2;
    cudaMalloc(&dA, szA); cudaMalloc(&dB, szB);
    cudaMalloc(&dC1, szC); cudaMalloc(&dC2, szC);
    cudaMemcpy(dA, hA, szA, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, szB, cudaMemcpyHostToDevice);

    dim3 block(TILE, TILE);
    dim3 grid((N+TILE_K-1)/TILE_K, (M+TILE_K-1)/TILE_K);

    float t1 = timeKernel(sgemm_stride4,         grid, block, dA,dB,dC1, M,N,K, "STRIDE=4");
    float t2 = timeKernel(sgemm_stride4_regtile, grid, block, dA,dB,dC2, M,N,K, "STRIDE=4 + RegTile");

    printf("\n  Speedup: %.2fx (%.2f vs %.2f ms)\n", t1/t2, t1, t2);

    cudaFree(dA); cudaFree(dB); cudaFree(dC1); cudaFree(dC2);
    free(hA); free(hB);
    return 0;
}
