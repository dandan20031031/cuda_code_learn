#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define TILE 16
#define STRIDE 2

// ============================================================
// Coarsening + K-tile also enlarged: TILE_K = TILE * STRIDE
// Block=16x16 threads, each computes 2x2 outputs, K-tile=32
// Shared memory: As[32][32] + Bs[32][32] = 8KB
// ============================================================
#define TILE_K (TILE * STRIDE)

__global__ void sgemm_coarsen(float* A, float* B, float* C, int M, int N, int K)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row0 = blockIdx.y * TILE_K + ty * STRIDE;
    int col0 = blockIdx.x * TILE_K + tx * STRIDE;

    __shared__ float As[TILE_K][TILE_K];
    __shared__ float Bs[TILE_K][TILE_K];

    float reg[STRIDE][STRIDE] = {0.0f};

    // t 直接就是 K 方向的偏移量，每次跳 TILE_K
    for (int t = 0; t < K; t += TILE_K) {

        // 每线程搬 STRIDE×STRIDE 个 A 和 B（共 8 个 float）
        for (int di = 0; di < STRIDE; di++) {
            int r = row0 + di;
            for (int dj = 0; dj < STRIDE; dj++) {
                int c = col0 + dj;
                // A[r][t + tx*STRIDE + dj]
                As[ty*STRIDE+di][tx*STRIDE+dj] = (r < M)
                    ? A[r*K + t + tx*STRIDE + dj] : 0.0f;
                // B[t + ty*STRIDE + di][c]
                Bs[ty*STRIDE+di][tx*STRIDE+dj] = (c < N)
                    ? B[(t + ty*STRIDE + di)*N + c] : 0.0f;
            }
        }

        __syncthreads();

        // 从 shared memory 读 TILE_K 次（32 次 vs 原来 16 次）
        for (int di = 0; di < STRIDE; di++)
            for (int dj = 0; dj < STRIDE; dj++)
                for (int k = 0; k < TILE_K; k++)
                    reg[di][dj] += As[ty*STRIDE+di][k] * Bs[k][tx*STRIDE+dj];

        __syncthreads();
    }

    for (int di = 0; di < STRIDE; di++)
        for (int dj = 0; dj < STRIDE; dj++) {
            int r = row0 + di, c = col0 + dj;
            if (r < M && c < N) C[r * N + c] = reg[di][dj];
        }
}

// ============================================================
// Original tiled kernel (STRIDE=1) for comparison
// ============================================================
__global__ void sgemm_tiled(float* A, float* B, float* C, int M, int N, int K)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE + ty;
    int col = blockIdx.x * TILE + tx;

    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    float sum = 0.0f;
    int n_tiles = K / TILE;
    for (int t = 0; t < n_tiles; t++) {
        As[ty][tx] = (row < M) ? A[row*K + t*TILE + tx] : 0.0f;
        Bs[ty][tx] = (col < N) ? B[(t*TILE+ty)*N + col] : 0.0f;
        __syncthreads();
        for (int k = 0; k < TILE; k++)
            sum += As[ty][k] * Bs[k][tx];
        __syncthreads();
    }
    if (row < M && col < N) C[row*N + col] = sum;
}

// ============================================================
void init(float* m, int n) { for(int i=0;i<n;i++) m[i]=(rand()%100)/100.0f; }

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
    printf("  %-10s  %.4f ms\n", label, ms);
    cudaEventDestroy(s); cudaEventDestroy(e);
    return ms;
}

int main()
{
    srand(42);
    int M = 4096, N = 4096, K = 4096;

    size_t szA = M*K*sizeof(float), szB = K*N*sizeof(float), szC = M*N*sizeof(float);
    printf("\n=== SGEMM Coarsening Benchmark: %dx%dx%d (STRIDE=%d) ===\n",M,N,K,STRIDE);

    float *hA=(float*)malloc(szA), *hB=(float*)malloc(szB);
    init(hA, M*K); init(hB, K*N);

    float *dA, *dB, *dC1, *dC2;
    cudaMalloc(&dA, szA); cudaMalloc(&dB, szB);
    cudaMalloc(&dC1, szC); cudaMalloc(&dC2, szC);
    cudaMemcpy(dA, hA, szA, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, szB, cudaMemcpyHostToDevice);

    dim3 block(TILE, TILE);
    dim3 g1((N+TILE-1)/TILE, (M+TILE-1)/TILE);                    // tiled
    dim3 g2((N+TILE*STRIDE-1)/(TILE*STRIDE), (M+TILE*STRIDE-1)/(TILE*STRIDE)); // coarsen

    printf("\nGPU time:\n");
    timeKernel(sgemm_tiled,   g1, block, dA,dB,dC1, M,N,K, "tiled");
    timeKernel(sgemm_coarsen, g2, block, dA,dB,dC2, M,N,K, "coarsen");

    printf("(less blocks = less launch overhead, more work per thread)\n");

    cudaFree(dA); cudaFree(dB); cudaFree(dC1); cudaFree(dC2);
    free(hA); free(hB);
    return 0;
}
