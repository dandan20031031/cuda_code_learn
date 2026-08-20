// Online Softmax V2：float4 向量化加载
__device__ void warpReduceOnline_v2(float& m, float& d) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        float m2 = __shfl_down_sync(0xffffffff, m, offset);
        float d2 = __shfl_down_sync(0xffffffff, d, offset);
        float m_new = fmaxf(m, m2);
        d = d * expf(m - m_new) + d2 * expf(m2 - m_new);
        m = m_new;
    }
}

__global__ void online_softmax_v2_float4(
    const float* input,    // M×N, row-major
    float* output,         // M×N, row-major
    int M, int N)
{
    int row = blockIdx.x;   // 一个 block 处理一行
    int tid = threadIdx.x;
    int lane = tid & 31;    // tid % 32
    int wid  = tid >> 5;    // tid / 32

    const float* x = input + row * N;
    float* y = output + row * N;

    // 行首指针转成 float4，计数单位从 N 变成 N4 = N/4
    const float4* x4 = reinterpret_cast<const float4*>(x);
    float4*       y4 = reinterpret_cast<float4*>(y);
    int N4 = N / 4;

    // 第 1 遍：向量化 Online 递推（一次读 4 个，逐个更新 (m, d)）
    float local_m = -INFINITY;
    float local_d = 0.0f;
    for (int i = tid; i < N4; i += blockDim.x) {
        float4 v = x4[i];
        float mn;
        mn = fmaxf(local_m, v.x); local_d = local_d * expf(local_m - mn) + expf(v.x - mn); local_m = mn;
        mn = fmaxf(local_m, v.y); local_d = local_d * expf(local_m - mn) + expf(v.y - mn); local_m = mn;
        mn = fmaxf(local_m, v.z); local_d = local_d * expf(local_m - mn) + expf(v.z - mn); local_m = mn;
        mn = fmaxf(local_m, v.w); local_d = local_d * expf(local_m - mn) + expf(v.w - mn); local_m = mn;
    }

    // 尾部：N 不是 4 的倍数时的剩余元素，用标量处理
    for (int i = N4 * 4 + tid; i < N; i += blockDim.x) {
        float xi = x[i];
        float m_new = fmaxf(local_m, xi);
        local_d = local_d * expf(local_m - m_new) + expf(xi - m_new);
        local_m = m_new;
    }

    // 两级 Warp Shuffle 合并（与 V1 相同）
    warpReduceOnline_v2(local_m, local_d);

    __shared__ float warp_m[32];
    __shared__ float warp_d[32];
    if (lane == 0) {
        warp_m[wid] = local_m;
        warp_d[wid] = local_d;
    }
    __syncthreads();

    int num_warps = blockDim.x / 32;
    if (wid == 0) {
        local_m = (lane < num_warps) ? warp_m[lane] : -INFINITY;
        local_d = (lane < num_warps) ? warp_d[lane] : 0.0f;
        warpReduceOnline_v2(local_m, local_d);
    }

    __shared__ float final_m, final_d;
    if (tid == 0) {
        final_m = local_m;
        final_d = local_d;
    }
    __syncthreads();

    float row_max = final_m;
    float inv_sum = 1.0f / final_d;

    // 第 2 遍：向量化归一化写出（构造 float4 一次 STG.128 写回）
    for (int i = tid; i < N4; i += blockDim.x) {
        float4 v = x4[i];
        float4 o;
        o.x = expf(v.x - row_max) * inv_sum;
        o.y = expf(v.y - row_max) * inv_sum;
        o.z = expf(v.z - row_max) * inv_sum;
        o.w = expf(v.w - row_max) * inv_sum;
        y4[i] = o;
    }
    for (int i = N4 * 4 + tid; i < N; i += blockDim.x) {
        y[i] = expf(x[i] - row_max) * inv_sum;
    }
}
