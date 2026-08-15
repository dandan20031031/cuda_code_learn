#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CUDA_CHK(call) do { \
    cudaError_t _e_ = (call); \
    if (_e_ != cudaSuccess) { \
        printf("CUDA ERROR at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e_)); \
        exit(1); \
    } \
} while(0)

#define CUBLAS_CHK(call) do { \
    cublasStatus_t _s_ = (call); \
    if (_s_ != CUBLAS_STATUS_SUCCESS) { \
        printf("CUBLAS ERROR at %s:%d: %d\n", __FILE__, __LINE__, (int)_s_); \
        exit(1); \
    } \
} while(0)

// FP32 SGEMM, 和你 ex1~ex9 完全一致的精度配置
void run_cublas_sgemm(cublasHandle_t handle,
                      const float *d_A, const float *d_B, float *d_C,
                      int M, int N, int K, float *ms_out)
{
    const float alpha = 1.0f;
    const float beta  = 0.0f;

    // 行优先 C(M×N)=A(M×K)×B(K×N)  等价于  列优先 C^T(N×M)=B^T(N×K)×A^T(K×M)
    // cuBLAS 只认列优先 → 交换 A/B + 交换 M/N（和 ex10 一样的 trick）
    cudaEvent_t start, stop;
    CUDA_CHK(cudaEventCreate(&start));
    CUDA_CHK(cudaEventCreate(&stop));

    // 预热
    CUBLAS_CHK(cublasSgemm(handle,
                           CUBLAS_OP_N, CUBLAS_OP_N,
                           N, M, K,
                           &alpha,
                           d_B, N,   // 列优先视角 = B^T (N×K)
                           d_A, K,   // 列优先视角 = A^T (K×M)
                           &beta,
                           d_C, N)); // 输出 (N×M) 列优先 = C^T
    CUDA_CHK(cudaDeviceSynchronize());

    CUDA_CHK(cudaEventRecord(start));
    CUBLAS_CHK(cublasSgemm(handle,
                           CUBLAS_OP_N, CUBLAS_OP_N,
                           N, M, K,
                           &alpha,
                           d_B, N,
                           d_A, K,
                           &beta,
                           d_C, N));
    CUDA_CHK(cudaEventRecord(stop));
    CUDA_CHK(cudaEventSynchronize(stop));

    CUDA_CHK(cudaEventElapsedTime(ms_out, start, stop));
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    const int M = 4096, N = 4096, K = 4096;

    printf("\n=== cuBLAS SGEMM (FP32) ===\n");
    printf("  M=N=K=%d\n", M);

    // 初始化数据
    float *hA = (float *)malloc(M * K * sizeof(float));
    float *hB = (float *)malloc(K * N * sizeof(float));
    for (int i = 0; i < M * K; i++) hA[i] = (rand() % 100) / 100.0f;
    for (int i = 0; i < K * N; i++) hB[i] = (rand() % 100) / 100.0f;

    float *dA, *dB, *dC;
    CUDA_CHK(cudaMalloc(&dA, M * K * sizeof(float)));
    CUDA_CHK(cudaMalloc(&dB, K * N * sizeof(float)));
    CUDA_CHK(cudaMalloc(&dC, M * N * sizeof(float)));
    CUDA_CHK(cudaMemcpy(dA, hA, M * K * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHK(cudaMemcpy(dB, hB, K * N * sizeof(float), cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    CUBLAS_CHK(cublasCreate(&handle));

    float ms;
    run_cublas_sgemm(handle, dA, dB, dC, M, N, K, &ms);

    double tflops = 2.0 * M * N * K / (ms / 1000.0) / 1e12;
    printf("  Time: %.4f ms  |  %.2f TFLOPS\n\n", ms, tflops);

    cublasDestroy(handle);
    free(hA); free(hB);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return 0;
}
