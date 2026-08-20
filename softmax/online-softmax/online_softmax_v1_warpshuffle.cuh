
__device__ void warpReduceOnline_v1(float& m, float& d) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        float m2 = __shfl_down_sync(0xffffffff, m, offset);
        float d2 = __shfl_down_sync(0xffffffff, d, offset);
        float m_new = fmaxf(m, m2);
        d = d * expf(m - m_new) + d2 * expf(m2 - m_new);
        m = m_new;
    }
}

__global__ void online_softmax_v1_warpshuffle(
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

    // 第 1 遍：每线程 Online 处理自己的一段，得到局部 (local_m, local_d)
    float local_m = -INFINITY;
    float local_d = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) {
        float xi = x[i];
        float m_new = fmaxf(local_m, xi);
        local_d = local_d * expf(local_m - m_new) + expf(xi - m_new);
        local_m = m_new;
    }

    // 第一级：warp 内合并
    warpReduceOnline_v1(local_m, local_d);

    // warp 间通过 Shared Memory 交换
    __shared__ float warp_m[32];
    __shared__ float warp_d[32];
    if (lane == 0) {
        warp_m[wid] = local_m;
        warp_d[wid] = local_d;
    }
    __syncthreads();

    // 第二级：warp 0 做最终合并
    int num_warps = blockDim.x / 32;
    if (wid == 0) {
        local_m = (lane < num_warps) ? warp_m[lane] : -INFINITY;
        local_d = (lane < num_warps) ? warp_d[lane] : 0.0f;
        warpReduceOnline_v1(local_m, local_d);
    }

    // 广播最终结果给 block 内所有线程
    __shared__ float final_m, final_d;
    if (tid == 0) {
        final_m = local_m;
        final_d = local_d;
    }
    __syncthreads();

    float row_max = final_m;
    float inv_sum = 1.0f / final_d;

    // 第 2 遍：用全局 (m, d) 归一化写出
    for (int i = tid; i < N; i += blockDim.x) {
        y[i] = expf(x[i] - row_max) * inv_sum;
    }
}
