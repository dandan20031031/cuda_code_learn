#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <math.h>

// V1：一个 Block 协作处理一行，用 Shared Memory 做并行规约
// 启动方式：dim3 block(block_size, 1); dim3 grid(1, M);
//          动态共享内存：<<<grid, block, block_size * sizeof(float)>>>
__global__ void softmax_v2_sharedreduce(
    float *input,    // M×N, row-major
    float *output,   // M×N, row-major
    int M, int N)
{
    int row = blockIdx.y;   // 一个 block 处理一行
    int tx = threadIdx.x;   // block 内的线程号
    if (row >= M) return;

    float *x = input + row * N;
    float *y = output + row * N;

    extern __shared__ float smem[];   // 动态共享内存，大小由启动时传入

    // Pass 1：并行求行最大值
    float max_val = -INFINITY;
    for (int i = tx; i < N; i += blockDim.x) {
        max_val = fmaxf(max_val, x[i]);
    }
    smem[tx] = max_val;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tx < s) {
            smem[tx] = fmaxf(smem[tx], smem[tx + s]);
        }
        __syncthreads();
    }
    max_val = smem[0];
    __syncthreads();

    // Pass 2：并行求指数和
    float sum = 0.0f;
    for (int i = tx; i < N; i += blockDim.x) {
        sum += expf(x[i] - max_val);
    }
    smem[tx] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tx < s) {
            smem[tx] += smem[tx + s];
        }
        __syncthreads();
    }
    sum = smem[0];
    __syncthreads();

    // Pass 3：归一化写出
    for (int i = tx; i < N; i += blockDim.x) {
        y[i] = expf(x[i] - max_val) / sum;
    }
}
