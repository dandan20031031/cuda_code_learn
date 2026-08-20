// Online Softmax V4：Grid Stride 多行并行
// 前面的版本都是「一个 block 处理一行」，grid = M。当 M 很大时，会启动海量 block：
//   - Grid 调度开销随 M 线性增加
//   - 每个 block 只跑一行就退出，smem / 寄存器状态无法跨行复用
// V4 用固定数量的 block 循环覆盖所有行（grid stride），算法层面与 V2（两遍 + float4）完全相同，
// 区别只在 block 与行的映射：V2 是 1:1，V4 是 1:多。
// 适用场景：M 很大。对 M 很小（如 batch=1）、N 很大的场景无效（那需要多 block 协作处理同一行）。
// 注意：grid-stride 循环对所有线程一致，循环内的 __syncthreads() 是安全的。
// 启动方式：grid = min(M, sm_count * blocks_per_sm)，block = block_size。

// 为避免与 V1/V2/V3 的同名辅助函数冲突，这里统一加 _v4 后缀
__device__ void warpReduceOnline_v4(float& m, float& d) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        float m2 = __shfl_down_sync(0xffffffff, m, offset);
        float d2 = __shfl_down_sync(0xffffffff, d, offset);
        float m_new = fmaxf(m, m2);
        d = d * expf(m - m_new) + d2 * expf(m2 - m_new);
        m = m_new;
    }
}

__global__ void online_softmax_v4_gridstride(
    const float* input,    // M×N, row-major
    float* output,         // M×N, row-major
    int M, int N)
{
    int tid = threadIdx.x;
    int lane = tid & 31;    // tid % 32
    int wid  = tid >> 5;    // tid / 32

    // 共享内存声明在循环外，跨行复用（同一 block 连续处理多行）
    __shared__ float warp_m[32];
    __shared__ float warp_d[32];
    __shared__ float final_m, final_d;

    for (int row = blockIdx.x; row < M; row += gridDim.x) {
        const float* x = input + row * N;
        float* y = output + row * N;

        const float4* x4 = reinterpret_cast<const float4*>(x);
        float4*       y4 = reinterpret_cast<float4*>(y);
        int N4 = N / 4;

        // 第 1 遍：向量化 Online 递推
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
        // 尾部（N 不是 4 的倍数）
        for (int i = N4 * 4 + tid; i < N; i += blockDim.x) {
            float xi = x[i];
            float m_new = fmaxf(local_m, xi);
            local_d = local_d * expf(local_m - m_new) + expf(xi - m_new);
            local_m = m_new;
        }

        // 两级 Warp Shuffle 合并
        warpReduceOnline_v4(local_m, local_d);
        if (lane == 0) {
            warp_m[wid] = local_m;
            warp_d[wid] = local_d;
        }
        __syncthreads();

        int num_warps = blockDim.x / 32;
        if (wid == 0) {
            local_m = (lane < num_warps) ? warp_m[lane] : -INFINITY;
            local_d = (lane < num_warps) ? warp_d[lane] : 0.0f;
            warpReduceOnline_v4(local_m, local_d);
        }
        if (tid == 0) {
            final_m = local_m;
            final_d = local_d;
        }
        __syncthreads();

        float row_max = final_m;
        float inv_sum = 1.0f / final_d;

        // 第 2 遍：向量化归一化写出
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
}
