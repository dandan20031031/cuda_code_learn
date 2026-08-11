#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define TILE 16
#define STRIDE 4
#define TILE_K (TILE * STRIDE)   // 64

// ============================================================
// STRIDE=4 Coarsening + float4 向量化加载
// ============================================================
__global__ void sgemm_coarsen4(float* A, float* B, float* C, int M, int N, int K)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row0 = blockIdx.y * TILE_K + ty * STRIDE;
    int col0 = blockIdx.x * TILE_K + tx * STRIDE;

    __shared__ float As[TILE_K][TILE_K];   // 64x64 = 16KB
    __shared__ float Bs[TILE_K][TILE_K];   // 64x64 = 16KB

    float reg[STRIDE][STRIDE] = {0.0f};

    for (int t = 0; t < K; t += TILE_K) {

        // ---- load A: each thread loads 1 row x 4 floats via float4 ----
        for (int di = 0; di < STRIDE; di++) {
            int r = row0 + di;
            int sm_row = ty * STRIDE + di;
            int sm_col = tx * STRIDE;

            if (r < M && t + sm_col + STRIDE - 1 < K) {
                float4 tmp = *(float4*)&A[r * K + t + sm_col];
                As[sm_row][sm_col + 0] = tmp.x;
                As[sm_row][sm_col + 1] = tmp.y;
                As[sm_row][sm_col + 2] = tmp.z;
                As[sm_row][sm_col + 3] = tmp.w;
            } else {
                // row or K out of bounds: manual zero-fill
                for (int dj = 0; dj < STRIDE; dj++)
                    As[sm_row][sm_col + dj] =
                        (r < M && t + sm_col + dj < K) ? A[r * K + t + sm_col + dj] : 0.0f;
            }
        }

        // ---- load B: each thread loads 1 row x 4 floats via float4 ----
        for (int di = 0; di < STRIDE; di++) {
            int sm_row = ty * STRIDE + di;
            int sm_col = tx * STRIDE;
            int kr = t + sm_row;

            if (col0 + STRIDE - 1 < N && kr < K) {
                float4 tmp = *(float4*)&B[kr * N + col0];
                Bs[sm_row][sm_col + 0] = tmp.x;
                Bs[sm_row][sm_col + 1] = tmp.y;
                Bs[sm_row][sm_col + 2] = tmp.z;
                Bs[sm_row][sm_col + 3] = tmp.w;
            } else {
                for (int dj = 0; dj < STRIDE; dj++)
                    Bs[sm_row][sm_col + dj] =
                        (kr < K && col0 + dj < N) ? B[kr * N + col0 + dj] : 0.0f;
            }
        }

        __syncthreads();

        // ---- compute: 4x4 x 64 = 1024 shared memory reads per thread ----
        for (int di = 0; di < STRIDE; di++)
            for (int dj = 0; dj < STRIDE; dj++)
                for (int k = 0; k < TILE_K; k++)
                    reg[di][dj] += As[ty * STRIDE + di][k]
                                 * Bs[k][tx * STRIDE + dj];

        __syncthreads();
    }

    // ---- write 4x4 = 16 results ----
    for (int di = 0; di < STRIDE; di++)
        for (int dj = 0; dj < STRIDE; dj++) {
            int r = row0 + di, c = col0 + dj;
            if (r < M && c < N) C[r * N + c] = reg[di][dj];
        }
}

// ============================================================
// STRIDE=2 version for comparison
// ============================================================
__global__ void sgemm_coarsen2(float* A, float* B, float* C, int M, int N, int K)
{
    const int S2 = 2, TK2 = TILE * S2;
    int tx = threadIdx.x, ty = threadIdx.y;
    int row0 = blockIdx.y * TK2 + ty * S2;
    int col0 = blockIdx.x * TK2 + tx * S2;

    __shared__ float As[TK2][TK2];
    __shared__ float Bs[TK2][TK2];
    float reg[S2][S2] = {0.0f};

    for (int t = 0; t < K; t += TK2) {
        for (int di = 0; di < S2; di++)
            for (int dj = 0; dj < S2; dj++) {
                int r = row0 + di, c = col0 + dj;
                As[ty*S2+di][tx*S2+dj] = (r < M) ? A[r*K + t + tx*S2 + dj] : 0.0f;
                Bs[ty*S2+di][tx*S2+dj] = (c < N) ? B[(t + ty*S2 + di)*N + c] : 0.0f;
            }
        __syncthreads();
        for (int di = 0; di < S2; di++)
            for (int dj = 0; dj < S2; dj++)
                for (int k = 0; k < TK2; k++)
                    reg[di][dj] += As[ty*S2+di][k] * Bs[k][tx*S2+dj];
        __syncthreads();
    }
    for (int di = 0; di < S2; di++)
        for (int dj = 0; dj < S2; dj++) {
            int r = row0+di, c = col0+dj;
            if (r<M && c<N) C[r*N + c] = reg[di][dj];
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
    printf("  %-12s  %.4f ms\n", label, ms);
    cudaEventDestroy(s); cudaEventDestroy(e);
    return ms;
}

void init(float* m, int n) { for(int i=0;i<n;i++) m[i]=(rand()%100)/100.0f; }

int main()
{
    srand(42);
    int M = 4096, N = 4096, K = 4096;

    size_t szA=M*K*sizeof(float), szB=K*N*sizeof(float), szC=M*N*sizeof(float);
    printf("\n=== SGEMM STRIDE=2 vs STRIDE=4 (float4): %dx%dx%d ===\n",M,N,K);

    float *hA=(float*)malloc(szA), *hB=(float*)malloc(szB);
    init(hA, M*K); init(hB, K*N);

    float *dA, *dB, *dC2, *dC4;
    cudaMalloc(&dA, szA); cudaMalloc(&dB, szB);
    cudaMalloc(&dC2, szC); cudaMalloc(&dC4, szC);
    cudaMemcpy(dA, hA, szA, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, szB, cudaMemcpyHostToDevice);

    dim3 block(TILE, TILE);
    dim3 g2((N+32-1)/32, (M+32-1)/32);   // STRIDE=2: TILE_K=32
    dim3 g4((N+64-1)/64, (M+64-1)/64);   // STRIDE=4: TILE_K=64

    float t2 = timeKernel(sgemm_coarsen2, g2, block, dA,dB,dC2, M,N,K, "STRIDE=2");
    float t4 = timeKernel(sgemm_coarsen4, g4, block, dA,dB,dC4, M,N,K, "STRIDE=4");

    printf("\n  Speedup: %.2fx (%.2f vs %.2f ms)\n", t2/t4, t2, t4);

    cudaFree(dA); cudaFree(dB); cudaFree(dC2); cudaFree(dC4);
    free(hA); free(hB);
    return 0;
}
