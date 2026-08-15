#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32

// ============================================================
// GPU kernel: tiled matrix multiply with shared memory
// ============================================================
__global__ void sgemm_tiled(float* A, float* B, float* C,
                             int M, int N, int K)
{
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;

    float sum = 0.0f;
    for (int t = 0; t < K; t += TILE_SIZE) {
        As[ty][tx] = A[row * K + (t + tx)];
        Bs[ty][tx] = B[(t + ty) * N + col];
        __syncthreads();
        for (int k = 0; k < TILE_SIZE; k++)
            sum += As[ty][k] * Bs[k][tx];
        __syncthreads();
    }
    if (row < M && col < N) C[row * N + col] = sum;
}

// ============================================================
void init(float* m, int n) { for(int i=0;i<n;i++) m[i]=(rand()%100)/100.0f; }

int main()
{
    srand(42);
    int M = 4096, N = 4096, K = 4096;
    size_t szA=M*K*sizeof(float), szB=K*N*sizeof(float), szC=M*N*sizeof(float);
    printf("\n=== ex4: Tiled SGEMM (TILE=%d) %dx%dx%d ===\n", TILE_SIZE, M,N,K);

    float *hA=(float*)malloc(szA), *hB=(float*)malloc(szB);
    init(hA, M*K); init(hB, K*N);

    float *dA, *dB, *dC;
    cudaMalloc(&dA, szA); cudaMalloc(&dB, szB); cudaMalloc(&dC, szC);
    cudaMemcpy(dA, hA, szA, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, szB, cudaMemcpyHostToDevice);

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N+TILE_SIZE-1)/TILE_SIZE, (M+TILE_SIZE-1)/TILE_SIZE);

    cudaEvent_t s, e;
    cudaEventCreate(&s); cudaEventCreate(&e);
    cudaEventRecord(s);
    sgemm_tiled<<<grid,block>>>(dA,dB,dC,M,N,K);
    cudaEventRecord(e);
    cudaEventSynchronize(e);
    float ms; cudaEventElapsedTime(&ms,s,e);
    printf("  GPU time: %.4f ms\n", ms);

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    free(hA); free(hB);
    cudaEventDestroy(s); cudaEventDestroy(e);
    return 0;
}
