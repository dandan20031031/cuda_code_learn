#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <math.h>

// V4：Warp Shuffle 两级规约 + float4 向量化加载/写出
// 相比 V3，三遍循环都用 float4（128-bit）一次性读写 4 个元素：
//   - 指令数更少（LDG.128 / STG.128）
//   - 合并访存更充分（4 个连续线程凑满 128B cache line）
// 归约部分与 V3 完全一致，瓶颈仍在访存，这一步直接降低访存指令开销。
// 启动方式：dim3 block(block_size, 1); dim3 grid(1, M);
// 前提：N 必须是 4 的倍数（8192 满足）；指针 16 字节对齐由 cudaMalloc 保证。

// 为避免与 V3 的同名辅助函数冲突，这里统一加 _v4 后缀
__device__ float warpReduceMax_v4(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

__device__ float warpReduceSum_v4(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ float blockReduceMax_v4(float val) {
    __shared__ float warp_max[32];
    int lane = threadIdx.x & 31;   // threadIdx.x % 32
    int wid  = threadIdx.x >> 5;   // threadIdx.x / 32

    val = warpReduceMax_v4(val);
    if (lane == 0) warp_max[wid] = val;
    __syncthreads();

    int nwarps = blockDim.x / 32;
    val = (lane < nwarps) ? warp_max[lane] : -INFINITY;
    if (wid == 0) val = warpReduceMax_v4(val);

    __shared__ float result;
    if (threadIdx.x == 0) result = val;
    __syncthreads();
    return result;
}

__device__ float blockReduceSum_v4(float val) {
    __shared__ float warp_sum[32];
    int lane = threadIdx.x & 31;
    int wid  = threadIdx.x >> 5;

    val = warpReduceSum_v4(val);
    if (lane == 0) warp_sum[wid] = val;
    __syncthreads();

    int nwarps = blockDim.x / 32;
    val = (lane < nwarps) ? warp_sum[lane] : 0.0f;
    if (wid == 0) val = warpReduceSum_v4(val);

    __shared__ float result;
    if (threadIdx.x == 0) result = val;
    __syncthreads();
    return result;
}

__global__ void softmax_v4_float4(
    float *input,    // M×N, row-major
    float *output,   // M×N, row-major
    int M, int N)
{
    int row = blockIdx.y;   // 一个 block 处理一行
    int tx  = threadIdx.x;  // block 内的线程号
    if (row >= M) return;

    // 行首指针转成 float4，计数单位从 N 变成 N/4
    const float4* x4 = reinterpret_cast<const float4*>(input + row * N);
    float4*       y4 = reinterpret_cast<float4*>(output + row * N);
    int N4 = N / 4;

    // Pass 1：求行最大值（一次读 4 个）
    float max_val = -INFINITY;
    for (int i = tx; i < N4; i += blockDim.x) {
        float4 v = x4[i];
        max_val = fmaxf(max_val, fmaxf(fmaxf(v.x, v.y), fmaxf(v.z, v.w)));
    }
    max_val = blockReduceMax_v4(max_val);

    // Pass 2：求指数和（4 个分量分别 exp 后累加）
    float sum = 0.0f;
    for (int i = tx; i < N4; i += blockDim.x) {
        float4 v = x4[i];
        sum += expf(v.x - max_val) + expf(v.y - max_val)
             + expf(v.z - max_val) + expf(v.w - max_val);
    }
    sum = blockReduceSum_v4(sum);

    // Pass 3：归一化写出（构造 float4 一次写回）
    float inv_sum = 1.0f / sum;
    for (int i = tx; i < N4; i += blockDim.x) {
        float4 v = x4[i];
        float4 o;
        o.x = expf(v.x - max_val) * inv_sum;
        o.y = expf(v.y - max_val) * inv_sum;
        o.z = expf(v.z - max_val) * inv_sum;
        o.w = expf(v.w - max_val) * inv_sum;
        y4[i] = o;
    }
}
