#ifndef NAIVE_ATTENTION_CUH
#define NAIVE_ATTENTION_CUH

#include <cuda_runtime.h>
#include <math.h>

// ============================================================
// Naive Attention（golden reference）
// ------------------------------------------------------------
// 每个线程负责一个 (b, h, q) query 行，三遍扫描（safe softmax）：
//   第1遍 找行最大值 m
//   第2遍 算分母 l = sum exp(S - m)
//   第3遍 算输出 O = sum (exp(S-m)/l * V)
//
// CAUSAL=true 时做因果 mask：第 q 行只能看 key 编号 <= q 的位置，
// 三遍循环里逐个 key 判断跳过（朴素做法，仅作 golden reference）
//
// 内存布局：[B, H, N, D] 连续，索引 = ((b*H + h)*N + n)*D + d
// HEAD_DIM 是模板参数（用于固定大小的局部数组 o[HEAD_DIM]）
// ============================================================
template<bool CAUSAL, int HEAD_DIM>
__global__ void naive_attention_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    int B, int H, int N)
{
    const int D = HEAD_DIM;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * H * N;
    if (tid >= total) return;

    int q  = tid % N;       // query 在该 (b,h) 内的行号
    int bh = tid / N;       // batch * head 的合并索引

    const float* Qq = Q + ((size_t)bh * N + q) * D;
    float*       Oq = O + ((size_t)bh * N + q) * D;
    float scale = 1.0f / sqrtf((float)D);

    // 第1遍：找行最大值 m
    float m = -INFINITY;
    for (int k = 0; k < N; ++k) {
        if (CAUSAL && k > q) break;   // key 编号超过行号就不可见，后面全不可见
        const float* Kk = K + ((size_t)bh * N + k) * D;
        float s = 0.0f;
        for (int d = 0; d < D; ++d) s += Qq[d] * Kk[d];
        s *= scale;
        if (s > m) m = s;
    }

    // 第2遍：算分母 l
    float l = 0.0f;
    for (int k = 0; k < N; ++k) {
        if (CAUSAL && k > q) break;
        const float* Kk = K + ((size_t)bh * N + k) * D;
        float s = 0.0f;
        for (int d = 0; d < D; ++d) s += Qq[d] * Kk[d];
        l += expf(s * scale - m);
    }

    // 第3遍：算输出 O = sum(p * V)
    float o[HEAD_DIM];
    for (int d = 0; d < D; ++d) o[d] = 0.0f;
    for (int k = 0; k < N; ++k) {
        if (CAUSAL && k > q) break;
        const float* Kk = K + ((size_t)bh * N + k) * D;
        const float* Vk = V + ((size_t)bh * N + k) * D;
        float s = 0.0f;
        for (int d = 0; d < D; ++d) s += Qq[d] * Kk[d];
        float p = expf(s * scale - m) / l;
        for (int d = 0; d < D; ++d) o[d] += p * Vk[d];
    }
    for (int d = 0; d < D; ++d) Oq[d] = o[d];
}

#endif
