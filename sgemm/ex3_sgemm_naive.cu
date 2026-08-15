#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void sgemm_naive(float* A, float* B, float* C,
                            int M, int N, int K)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++)
            sum += A[row * K + k] * B[k * N + col];
        C[row * N + col] = sum;
    }
}

void init(float* m, int n) { for(int i=0;i<n;i++) m[i]=(rand()%100)/100.0f; }

int main()
{
    srand(42);
    int M = 4096, N = 4096, K = 4096;
    size_t szA=M*K*sizeof(float), szB=K*N*sizeof(float), szC=M*N*sizeof(float);
    printf("\n=== ex3: Naive SGEMM %dx%dx%d ===\n", M,N,K);

    float *hA=(float*)malloc(szA), *hB=(float*)malloc(szB);
    init(hA, M*K); init(hB, K*N);

    float *dA, *dB, *dC;
    cudaMalloc(&dA, szA); cudaMalloc(&dB, szB); cudaMalloc(&dC, szC);
    cudaMemcpy(dA, hA, szA, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, szB, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((N+15)/16, (M+15)/16);

    cudaEvent_t s, e;
    cudaEventCreate(&s); cudaEventCreate(&e);
    cudaEventRecord(s);
    sgemm_naive<<<grid,block>>>(dA,dB,dC,M,N,K);
    cudaEventRecord(e);
    cudaEventSynchronize(e);
    float ms; cudaEventElapsedTime(&ms,s,e);
    printf("  GPU time: %.4f ms\n", ms);

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    free(hA); free(hB);
    cudaEventDestroy(s); cudaEventDestroy(e);
    return 0;
}
