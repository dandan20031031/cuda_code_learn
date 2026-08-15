#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// C(M×N) = A(M×K) × B(K×N)
int matrixMultiply()
{
    const int M = 2, K = 3, N = 4;

    // A: M×K = 2×3
    int size_A = M * K;
    float *h_A = (float *)malloc(sizeof(float) * size_A);
    for (int i = 0; i < size_A; i++) h_A[i] = i + 1;

    // B: K×N = 3×4
    int size_B = K * N;
    float *h_B = (float *)malloc(sizeof(float) * size_B);
    for (int i = 0; i < size_B; i++) h_B[i] = i + 1;

    // C: M×N = 2×4
    int size_C = M * N;
    float *h_C = (float *)malloc(sizeof(float) * size_C);

    // GPU 显存
    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, sizeof(float) * size_A);
    cudaMalloc((void **)&d_B, sizeof(float) * size_B);
    cudaMalloc((void **)&d_C, sizeof(float) * size_C);
    cudaMemcpy(d_A, h_A, sizeof(float) * size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeof(float) * size_B, cudaMemcpyHostToDevice);

    const float alpha = 1.0f;
    const float beta  = 0.0f;

    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSgemm(handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &alpha, d_B, N,
        d_A, K,
        &beta, d_C, N);
    cublasDestroy(handle);

    cudaMemcpy(h_C, d_C, sizeof(float) * size_C, cudaMemcpyDeviceToHost);

    printf("\nC(%dx%d) =\n", M, N);
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
            printf("%5.1f  ", h_C[i * N + j]);
        printf("\n");
    }

    free(h_A); free(h_B); free(h_C);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    return 0;
}

int main()
{
    matrixMultiply();
    getchar();
    return 0;
}
