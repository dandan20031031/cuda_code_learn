// softmax_bench.cu —— Softmax kernel 统一基准测试
// 编译（配合现有 tasks.json，对当前文件构建即可）：
//   nvcc softmax_bench.cu -o softmax_bench.exe -arch=sm_120 -Xcompiler /utf-8
// 每个版本的 kernel 放在独立 .cuh 里，在这里 include 后统一对拍 + 计时。

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "softmax_v1_naive.cuh"
#include "softmax_v2_sharedreduce.cuh"
#include "softmax_v3_warpshuffle.cuh"
#include "softmax_v4_float4.cuh"
#include "softmax_v5_twopass.cuh"

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d -- %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err));                                   \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

// CPU 参考实现（Safe Softmax：先减行最大值再 exp），用于对拍
void softmax_cpu(const float* input, float* output, int M, int N) {
    for (int row = 0; row < M; row++) {
        const float* x = input + row * N;
        float* y = output + row * N;

        // 1) 求行最大值
        float max_val = -INFINITY;
        for (int k = 0; k < N; k++) {
            if (x[k] > max_val) max_val = x[k];
        }

        // 2) 求指数和
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += expf(x[k] - max_val);
        }

        // 3) 归一化
        float inv_sum = 1.0f / sum;
        for (int k = 0; k < N; k++) {
            y[k] = expf(x[k] - max_val) * inv_sum;
        }
    }
}

// ---------- 各版本的启动封装（host 函数，供统一计时循环调用） ----------

// V1 naive：一个线程处理一行
void launch_v1_naive(float* d_input, float* d_output, int M, int N) {
    int block_size = 256;
    dim3 block(1, block_size);                      // x 方向 1 个线程
    dim3 grid(1, (M + block_size - 1) / block_size); // 行号由 y 方向产生
    softmax_v1_naive<<<grid, block>>>(d_input, d_output, M, N);
}

// V2 shared reduce：一个 block 处理一行，block 内多线程 + 共享内存规约
void launch_v2_sharedreduce(float* d_input, float* d_output, int M, int N) {
    int block_size = 256;                           // 必须为 2 的幂（树形规约前提）
    dim3 block(block_size, 1);                      // 256 个线程在 x 方向协作
    dim3 grid(1, M);                                // 每个 block 处理一行
    softmax_v2_sharedreduce<<<grid, block, block_size * sizeof(float)>>>(
        d_input, d_output, M, N);
}

// V3 warp shuffle：一个 block 处理一行，warp 内 shuffle + warp 间 smem 两级规约
void launch_v3_warpshuffle(float* d_input, float* d_output, int M, int N) {
    int block_size = 256;                            // 必须是 32 的倍数
    dim3 block(block_size, 1);                       // 256 个线程在 x 方向
    dim3 grid(1, M);                                 // 每个 block 处理一行
    softmax_v3_warpshuffle<<<grid, block>>>(d_input, d_output, M, N);
}

// V4 float4：V3 基础上，三遍循环都用 float4 向量化读写
void launch_v4_float4(float* d_input, float* d_output, int M, int N) {
    int block_size = 256;                            // 必须是 32 的倍数
    dim3 block(block_size, 1);                       // 256 个线程在 x 方向
    dim3 grid(1, M);                                 // 每个 block 处理一行
    softmax_v4_float4<<<grid, block>>>(d_input, d_output, M, N);
}

// V5 两遍融合：online softmax，把 max/sum 合并到一次读取
void launch_v5_twopass(float* d_input, float* d_output, int M, int N) {
    int block_size = 256;                            // 必须是 32 的倍数
    dim3 block(block_size, 1);                       // 256 个线程在 x 方向
    dim3 grid(1, M);                                 // 每个 block 处理一行
    softmax_v5_twopass<<<grid, block>>>(d_input, d_output, M, N);
}

// 统一计时 + 对拍 + 打印
typedef void (*launch_fn)(float*, float*, int, int);

void run_bench(const char* name, launch_fn launch,
               float* d_input, float* d_output, float* h_output, const float* h_ref,
               int M, int N, cudaEvent_t start, cudaEvent_t stop) {
    const size_t bytes = (size_t)M * N * sizeof(float);

    // 预热一次（避免首次启动开销混入计时）
    launch(d_input, d_output, M, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    // 正式计时
    CUDA_CHECK(cudaEventRecord(start));
    launch(d_input, d_output, M, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    // 结果拷回 host 并找最大误差
    CUDA_CHECK(cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost));
    float max_err = 0.0f;
    for (int i = 0; i < M * N; i++) {
        float err = fabsf(h_output[i] - h_ref[i]);
        if (err > max_err) max_err = err;
    }

    // 有效带宽：DRAM 最小流量 = 读 1 次输入 + 写 1 次输出 = 2 * M * N * 4 字节
    // （重复读命中 L2，不计入 DRAM；这是可与理论峰值对比的诚实口径）
    double total_bytes = 2.0 * (double)bytes;
    double gbps = total_bytes / (ms / 1000.0) / 1e9;

    printf("%-22s | %8.3f ms | %8.2f GB/s | err %e | %s\n",
           name, ms, gbps, max_err, (max_err < 1e-4f) ? "PASS" : "FAIL");
}

int main(int argc, char** argv) {
    // 矩阵尺寸可配置：./softmax_bench.exe [M] [N]，默认 4096×4096
    // 只给一个参数时 M = N（方阵）
    int M = (argc > 1) ? atoi(argv[1]) : 4096;   // 行数（如序列长度）
    int N = (argc > 2) ? atoi(argv[2]) : M;      // 列数（每行元素个数）
    const size_t bytes = (size_t)M * N * sizeof(float);

    // 分配 host / device 内存
    float* h_input  = (float*)malloc(bytes);
    float* h_output = (float*)malloc(bytes);
    float* h_ref    = (float*)malloc(bytes);
    if (!h_input || !h_output || !h_ref) {
        fprintf(stderr, "host malloc failed\n");
        return EXIT_FAILURE;
    }

    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, bytes));
    CUDA_CHECK(cudaMalloc(&d_output, bytes));

    // 初始化输入（固定随机种子保证可复现；范围 [-10, 10)）
    srand(42);
    for (int i = 0; i < M * N; i++) {
        h_input[i] = (rand() % 200 - 100) * 0.1f;
    }

    // CPU 参考结果
    softmax_cpu(h_input, h_ref, M, N);

    // 拷贝输入到 device
    CUDA_CHECK(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));

    // 计时用事件
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    printf("Softmax benchmark (M=%d, N=%d)\n", M, N);
    printf("%-22s | %10s | %12s | %s\n", "kernel", "time", "bandwidth", "result");
    printf("--------------------------------------------------------------\n");

    run_bench("v1_naive",        launch_v1_naive,        d_input, d_output, h_output, h_ref, M, N, start, stop);
    run_bench("v2_sharedreduce", launch_v2_sharedreduce, d_input, d_output, h_output, h_ref, M, N, start, stop);
    run_bench("v3_warpshuffle", launch_v3_warpshuffle, d_input, d_output, h_output, h_ref, M, N, start, stop);
    run_bench("v4_float4",       launch_v4_float4,       d_input, d_output, h_output, h_ref, M, N, start, stop);
    run_bench("v5_twopass",      launch_v5_twopass,      d_input, d_output, h_output, h_ref, M, N, start, stop);

    // 清理
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    free(h_input);
    free(h_output);
    free(h_ref);
    return 0;
}
