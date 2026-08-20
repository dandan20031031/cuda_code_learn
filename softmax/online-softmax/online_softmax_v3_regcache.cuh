// Online Softmax V3：寄存器缓存 One-Pass
__device__ void warpReduceOnline_v3(float& m, float& d) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        float m2 = __shfl_down_sync(0xffffffff, m, offset);
        float d2 = __shfl_down_sync(0xffffffff, d, offset);
        float m_new = fmaxf(m, m2);
        d = d * expf(m - m_new) + d2 * expf(m2 - m_new);
        m = m_new;
    }
}

template <int ELEMS>
__global__ void online_softmax_v3_regcache(
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

    // 每线程的寄存器缓存（ELEMS 是编译期常量）
    float reg_cache[ELEMS];

    // 第 1 遍（唯一一遍全局读取）：读 + 缓存 + Online 递推，完全展开
    float local_m = -INFINITY;
    float local_d = 0.0f;
    #pragma unroll
    for (int k = 0; k < ELEMS; k++) {
        int i = tid + k * blockDim.x;                 // 跨步访问，访存合并
        float xi = (i < N) ? x[i] : -INFINITY;        // 越界填充 -inf，不污染结果
        reg_cache[k] = xi;                            // 下标 k 为编译期常量 → 寄存器
        float m_new = fmaxf(local_m, xi);
        local_d = local_d * expf(local_m - m_new) + expf(xi - m_new);
        local_m = m_new;
    }

    // 两级 Warp Shuffle 合并（与 V1 相同）
    warpReduceOnline_v3(local_m, local_d);

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
        warpReduceOnline_v3(local_m, local_d);
    }

    __shared__ float final_m, final_d;
    if (tid == 0) {
        final_m = local_m;
        final_d = local_d;
    }
    __syncthreads();

    float row_max = final_m;
    float inv_sum = 1.0f / final_d;

    // 第 2 遍：从寄存器缓存读 + 写出（不再读全局内存），完全展开
    #pragma unroll
    for (int k = 0; k < ELEMS; k++) {
        int i = tid + k * blockDim.x;
        if (i < N) {
            y[i] = expf(reg_cache[k] - row_max) * inv_sum;
        }
    }
}
