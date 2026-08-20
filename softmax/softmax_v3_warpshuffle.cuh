#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <math.h>

// V3：Warp Shuffle 两级规约
// warp 内用 __shfl_down_sync 归约（免同步、寄存器级），warp 间用小块 smem + 1 次同步
// 启动方式：dim3 block(block_size, 1); dim3 grid(1, M);
// 注意：block_size 必须是 32 的倍数（保证每个 warp 满员，shuffle 的 0xffffffff 才安全）

// warp 内求最大值
__device__ float warpReduceMax(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

// warp 内求和
__device__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// block 内求最大值：第一级 warp 内 shuffle + 第二级 warp 间 smem
__device__ float blockReduceMax(float val) {
    __shared__ float warp_max[32];
    int lane = threadIdx.x & 31;   // threadIdx.x % 32
    int wid  = threadIdx.x >> 5;   // threadIdx.x / 32

    val = warpReduceMax(val);                // ① warp 内，无同步
    if (lane == 0) warp_max[wid] = val;
    __syncthreads();                         // ② 收集各 warp 的局部结果

    int nwarps = blockDim.x / 32;
    val = (lane < nwarps) ? warp_max[lane] : -INFINITY;
    if (wid == 0) val = warpReduceMax(val);  // ③ 第 0 个 warp 收尾

    __shared__ float result;
    if (threadIdx.x == 0) result = val;
    __syncthreads();                         // ④ 广播结果
    return result;
}

// block 内求和：同 blockReduceMax，把 fmaxf 换成 +
__device__ float blockReduceSum(float val) {
    __shared__ float warp_sum[32];
    int lane = threadIdx.x & 31;
    int wid  = threadIdx.x >> 5;

    val = warpReduceSum(val);
    if (lane == 0) warp_sum[wid] = val;
    __syncthreads();

    int nwarps = blockDim.x / 32;
    val = (lane < nwarps) ? warp_sum[lane] : 0.0f;
    if (wid == 0) val = warpReduceSum(val);

    __shared__ float result;
    if (threadIdx.x == 0) result = val;
    __syncthreads();
    return result;
}

__global__ void softmax_v3_warpshuffle(
    float *input,    // M×N, row-major
    float *output,   // M×N, row-major
    int M, int N)
{
    int row = blockIdx.y;   // 一个 block 处理一行
    int tx  = threadIdx.x;  // block 内的线程号
    if (row >= M) return;

    float *x = input + row * N;
    float *y = output + row * N;

    // Pass 1：求行最大值
    float max_val = -INFINITY;
    for (int i = tx; i < N; i += blockDim.x) {
        max_val = fmaxf(max_val, x[i]);
    }
    max_val = blockReduceMax(max_val);

    // Pass 2：求指数和
    float sum = 0.0f;
    for (int i = tx; i < N; i += blockDim.x) {
        sum += expf(x[i] - max_val);
    }
    sum = blockReduceSum(sum);

    // Pass 3：归一化写出
    for (int i = tx; i < N; i += blockDim.x) {
        y[i] = expf(x[i] - max_val) / sum;
    }
}
