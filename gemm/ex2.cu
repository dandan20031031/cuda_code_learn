#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>

// 错误检查宏
#define CHECK(call) {                                           \
    cudaError_t err = call;                                     \
    if (err != cudaSuccess) {                                  \
        printf("ERROR: %s:%d -> %s\n", __FILE__, __LINE__,     \
               cudaGetErrorString(err));                        \
        exit(-1);                                              \
    }                                                           \
}

__global__ void kernel(float* a, float* b, float* c, int N)
{
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < N) {
        c[id] = a[id] + b[id];
    }
}

void initData(float* addr, int N)
{
    for (int i = 0; i < N; i++) {
        addr[i] = i * 1.0f;
    }
}

int main()
{
    /* 1. 查询 GPU 设备信息 */
    int iDeviceCount = 0;
    CHECK(cudaGetDeviceCount(&iDeviceCount));
    printf("Device Count: %d\n\n", iDeviceCount);

    for (int i = 0; i < iDeviceCount; i++) {
        cudaDeviceProp prop;
        CHECK(cudaGetDeviceProperties(&prop, i));

        printf("GPU %d: %s\n", i, prop.name);
        printf("  计算能力:    %d.%d\n", prop.major, prop.minor);
        printf("  SM 数量:     %d\n", prop.multiProcessorCount);
        printf("  显存:        %.0f MB\n", (float)prop.totalGlobalMem / (1024 * 1024));
        printf("  Max Thread/Block: %d\n", prop.maxThreadsPerBlock);
        printf("  Max Block Dim:    (%d, %d, %d)\n",
               prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("  Max Grid Dim:     (%d, %d, %d)\n",
               prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
        printf("  Shared Mem/Block: %zu KB\n", prop.sharedMemPerBlock / 1024);
        printf("  Warp Size:        %d\n", prop.warpSize);
        printf("  时钟频率:         %.2f GHz\n", (float)prop.clockRate / 1e6);
        printf("\n");
    }

    CHECK(cudaSetDevice(0));

    /* 2. 分配内存 */
    int N = 512;
    size_t size = N * sizeof(float);

    float* h_a = (float*)malloc(size);
    float* h_b = (float*)malloc(size);
    float* h_c = (float*)malloc(size);

    float *d_a, *d_b, *d_c;
    CHECK(cudaMalloc(&d_a, size));
    CHECK(cudaMalloc(&d_b, size));
    CHECK(cudaMalloc(&d_c, size));

    /* 3. 初始化 CPU 数据 */
    initData(h_a, N);
    initData(h_b, N);

    /* 4. 拷贝到 GPU */
    CHECK(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice));

    /* 5. 执行 kernel */
    dim3 block(32);
    dim3 grid((N + block.x - 1) / block.x);

    cudaEvent_t start, stop;                    // 声明两个"时间戳"变量
    cudaEventCreate(&start);                   // 在 GPU 上创建一个时间戳标记
    cudaEventCreate(&stop);

    cudaEventRecord(start);                    // 记录当前 GPU 时间到 start
    kernel<<<grid, block>>>(d_a, d_b, d_c, N); // kernel 在 GPU 上排队执行
    cudaEventRecord(stop);                     // kernel 后面记录 stop（也是排队）

    CHECK(cudaGetLastError());                 // 检查 kernel 启动参数有没有错
    cudaEventSynchronize(stop);               // 等 stop 事件在 GPU 上完成
                                               // → 意味着 kernel 一定跑完了

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);    // 计算 start→stop 的 GPU 时间差(ms)
    printf("Kernel time: %.4f ms\n", ms);

    cudaEventDestroy(start);                   // 释放时间戳资源
    cudaEventDestroy(stop);
    CHECK(cudaDeviceSynchronize());

    /* 6. 结果拷回 CPU */
    CHECK(cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost));

    /* 7. 验证 */
    for (int i = 0; i < 5; i++) {
        printf("a[%d]=%.0f, b[%d]=%.0f, c[%d]=%.0f\n",
               i, h_a[i], i, h_b[i], i, h_c[i]);
    }

    /* 8. 释放 */
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    free(h_a); free(h_b); free(h_c);

    return 0;
}
