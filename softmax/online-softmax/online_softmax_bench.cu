// online_softmax_bench.cu —— Online Softmax 统一基准测试
// 编译（配合 cuda_code/.vscode/tasks.json，对当前文件构建即可）：
//   nvcc online_softmax_bench.cu -o online_softmax_bench.exe -arch=sm_120 -Xcompiler /utf-8
// 每个版本放在独立 .cuh 里，在这里 include 后统一对拍 + 计时。

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "online_softmax_v0_naive.cuh"
#include "online_softmax_v1_warpshuffle.cuh"
#include "online_softmax_v2_float4.cuh"
#include "online_softmax_v3_regcache.cuh"
#include "online_softmax_v4_gridstride.cuh"

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d -- %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err));                                   \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

// CPU 参考实现（Safe Softmax：先减行最大值再 exp），用于对拍。
// Online Softmax 与 Safe Softmax 在数学上等价，误差应在浮点精度范围内。
void softmax_cpu(const float* input, float* output, int M, int N) {
    for (int row = 0; row < M; row++) {
        const float* x = input + row * N;
        float* y = output + row * N;

        float max_val = -INFINITY;
        for (int k = 0; k < N; k++) {
            if (x[k] > max_val) max_val = x[k];
        }

        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += expf(x[k] - max_val);
        }

        float inv_sum = 1.0f / sum;
        for (int k = 0; k < N; k++) {
            y[k] = expf(x[k] - max_val) * inv_sum;
        }
    }
}

// ---------- 各版本启动封装（host 函数，供统一计时循环调用） ----------

// V0 naive：一个线程处理一行
void launch_v0_naive(float* d_input, float* d_output, int M, int N) {
    int block_size = 256;
    dim3 block(1, block_size);                       // x 方向 1 个线程
    dim3 grid(1, (M + block_size - 1) / block_size); // 行号由 y 方向产生
    online_softmax_v0_naive<<<grid, block>>>(d_input, d_output, M, N);
}

// V1 warpshuffle：一个 block 处理一行，warp shuffle 合并规约
void launch_v1_warpshuffle(float* d_input, float* d_output, int M, int N) {
    int block_size = 256;                            // 必须是 32 的倍数
    dim3 block(block_size, 1);                       // 256 个线程在 x 方向协作
    dim3 grid(M, 1);                                 // 每个 block 处理一行
    online_softmax_v1_warpshuffle<<<grid, block>>>(d_input, d_output, M, N);
}

// V2 float4：V1 基础上，两遍循环都用 float4 向量化读写
void launch_v2_float4(float* d_input, float* d_output, int M, int N) {
    int block_size = 256;                            // 必须是 32 的倍数
    dim3 block(block_size, 1);                       // 256 个线程在 x 方向协作
    dim3 grid(M, 1);                                 // 每个 block 处理一行
    online_softmax_v2_float4<<<grid, block>>>(d_input, d_output, M, N);
}

// V3 regcache：寄存器缓存 one-pass（全局只读 1 次），编译期展开
// 按每线程元素数分发到对应 ELEMS 的模板实例；超出模板范围或 N<block 时退回 V2
void launch_v3_regcache(float* d_input, float* d_output, int M, int N) {
    int block_size = 256;                            // 必须是 32 的倍数
    dim3 block(block_size, 1);
    dim3 grid(M, 1);
    int elems = (N + block_size - 1) / block_size;   // 每线程元素数（向上取整）
    if (N < block_size) {
        online_softmax_v2_float4<<<grid, block>>>(d_input, d_output, M, N);
    } else if (elems <= 16) {
        online_softmax_v3_regcache<16><<<grid, block>>>(d_input, d_output, M, N);
    } else if (elems <= 32) {
        online_softmax_v3_regcache<32><<<grid, block>>>(d_input, d_output, M, N);
    } else if (elems <= 64) {
        online_softmax_v3_regcache<64><<<grid, block>>>(d_input, d_output, M, N);
    } else {
        online_softmax_v2_float4<<<grid, block>>>(d_input, d_output, M, N);
    }
}

// V4 gridstride：固定 block 数循环覆盖所有行（grid = min(M, sm_count * 4)）
void launch_v4_gridstride(float* d_input, float* d_output, int M, int N) {
    int block_size = 256;                            // 必须是 32 的倍数
    dim3 block(block_size, 1);

    static int sm_count = 0;
    if (sm_count == 0) {
        cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0);
    }

    int grid_size = sm_count * 4;                    // 每 SM 4 个 block
    if (grid_size > M) grid_size = M;
    dim3 grid(grid_size, 1);

    online_softmax_v4_gridstride<<<grid, block>>>(d_input, d_output, M, N);
}

// 统一计时 + 对拍 + 打印。
// num_reads：该 kernel 对 input 的全局读取遍数（用于估算实际 DRAM 流量）。
//   - naive/两遍版：读 2 次 + 写 1 次
//   - 后续 one-pass 版：读 1 次 + 写 1 次
typedef void (*launch_fn)(float*, float*, int, int);

void run_bench(const char* name, launch_fn launch, int num_reads,
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

    // 有效带宽：实际全局内存流量 = (num_reads 次读 + 1 次写) * bytes
    double total_bytes = (double)(num_reads + 1) * (double)bytes;
    double gbps = total_bytes / (ms / 1000.0) / 1e9;

    printf("%-22s | %8.3f ms | %8.2f GB/s | err %e | %s\n",
           name, ms, gbps, max_err, (max_err < 1e-4f) ? "PASS" : "FAIL");
}

int main(int argc, char** argv) {
    // 矩阵尺寸可配置：./online_softmax_bench.exe [M] [N]，默认 4096×4096
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

    printf("Online softmax benchmark (M=%d, N=%d)\n", M, N);
    printf("%-22s | %10s | %12s | %s\n", "kernel", "time", "bandwidth", "result");
    printf("--------------------------------------------------------------\n");

    run_bench("v0_naive",        launch_v0_naive,        2, d_input, d_output, h_output, h_ref, M, N, start, stop);
    run_bench("v1_warpshuffle",  launch_v1_warpshuffle,  2, d_input, d_output, h_output, h_ref, M, N, start, stop);
    run_bench("v2_float4",       launch_v2_float4,       2, d_input, d_output, h_output, h_ref, M, N, start, stop);
    run_bench("v3_regcache",     launch_v3_regcache,     1, d_input, d_output, h_output, h_ref, M, N, start, stop);
    run_bench("v4_gridstride",   launch_v4_gridstride,   2, d_input, d_output, h_output, h_ref, M, N, start, stop);

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
