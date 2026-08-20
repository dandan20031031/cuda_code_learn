#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <math.h>

// V5：两遍融合（Online Softmax）+ float4 向量化
// 相比 V4（三遍：max / sum / 写出），把求 max 和求 sum 合并到同一次读取里：
//   - Pass 1：读一遍 input，在线更新 running (m, s)，再 block 归约出全局 max / sum
//   - Pass 2：读一遍 input，用全局 max / sum 归一化写出
// 这样 input 只读 2 次（V4 读 3 次）。对不命中 L2 的大矩阵，DRAM 流量从 4× 降到 3×，
// 理论上限约 1.33×。
// 关键：online 更新在遇到更大的 max 时，用 exp(m - m_new) 把旧 sum 重新缩放。
// 启动方式：dim3 block(block_size, 1); dim3 grid(1, M);
// 前提：N 必须是 4 的倍数。

// 归约辅助函数（加 _v5 后缀避免与其他版本重名）
__device__ float warpReduceMax_v5(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

__device__ float warpReduceSum_v5(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ float blockReduceMax_v5(float val) {
    __shared__ float warp_max[32];
    int lane = threadIdx.x & 31;   // threadIdx.x % 32
    int wid  = threadIdx.x >> 5;   // threadIdx.x / 32

    val = warpReduceMax_v5(val);
    if (lane == 0) warp_max[wid] = val;
    __syncthreads();

    int nwarps = blockDim.x / 32;
    val = (lane < nwarps) ? warp_max[lane] : -INFINITY;
    if (wid == 0) val = warpReduceMax_v5(val);

    __shared__ float result;
    if (threadIdx.x == 0) result = val;
    __syncthreads();
    return result;
}

__device__ float blockReduceSum_v5(float val) {
    __shared__ float warp_sum[32];
    int lane = threadIdx.x & 31;
    int wid  = threadIdx.x >> 5;

    val = warpReduceSum_v5(val);
    if (lane == 0) warp_sum[wid] = val;
    __syncthreads();

    int nwarps = blockDim.x / 32;
    val = (lane < nwarps) ? warp_sum[lane] : 0.0f;
    if (wid == 0) val = warpReduceSum_v5(val);

    __shared__ float result;
    if (threadIdx.x == 0) result = val;
    __syncthreads();
    return result;
}

__global__ void softmax_v5_twopass(
    float *input,    // M×N, row-major
    float *output,   // M×N, row-major
    int M, int N)
{
    int row = blockIdx.y;   // 一个 block 处理一行
    int tx  = threadIdx.x;  // block 内的线程号
    if (row >= M) return;

    const float4* x4 = reinterpret_cast<const float4*>(input + row * N);
    float4*       y4 = reinterpret_cast<float4*>(output + row * N);
    int N4 = N / 4;

    // Pass 1：一遍读取，在线更新 running (m, s)
    float m = -INFINITY;
    float s = 0.0f;
    for (int i = tx; i < N4; i += blockDim.x) {
        float4 v = x4[i];
        float mn;
        mn = fmaxf(m, v.x); s = s * expf(m - mn) + expf(v.x - mn); m = mn;
        mn = fmaxf(m, v.y); s = s * expf(m - mn) + expf(v.y - mn); m = mn;
        mn = fmaxf(m, v.z); s = s * expf(m - mn) + expf(v.z - mn); m = mn;
        mn = fmaxf(m, v.w); s = s * expf(m - mn) + expf(v.w - mn); m = mn;
    }

    // block 内归约：先求全局 max，再把自己的 s 缩放到全局 max 后求和
    float global_m = blockReduceMax_v5(m);
    s = s * expf(m - global_m);   // rescale：旧基准 m 下的 sum -> 全局 m 下的 sum
    float global_s = blockReduceSum_v5(s);

    // Pass 2：一遍读取，归一化写出
    float inv_sum = 1.0f / global_s;
    for (int i = tx; i < N4; i += blockDim.x) {
        float4 v = x4[i];
        float4 o;
        o.x = expf(v.x - global_m) * inv_sum;
        o.y = expf(v.y - global_m) * inv_sum;
        o.z = expf(v.z - global_m) * inv_sum;
        o.w = expf(v.w - global_m) * inv_sum;
        y4[i] = o;
    }
}
