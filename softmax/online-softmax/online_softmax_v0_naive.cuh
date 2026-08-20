// Online Softmax V0：单线程处理一行（naive 基线）
__global__ void online_softmax_v0_naive(
    const float* input,    // M×N, row-major
    float* output,         // M×N, row-major
    int M, int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= M) return;

    const float* x = input + row * N;
    float* y = output + row * N;

    // 第 1 遍：一遍读取，在线同时更新 (m, d)
    float m = -INFINITY;
    float d = 0.0f;
    for (int i = 0; i < N; i++) {
        float xi = x[i];
        float m_new = fmaxf(m, xi);
        d = d * expf(m - m_new) + expf(xi - m_new);
        m = m_new;
    }

    // 第 2 遍：用全局 (m, d) 归一化写出
    float inv_d = 1.0f / d;
    for (int i = 0; i < N; i++) {
        y[i] = expf(x[i] - m) * inv_d;
    }
}
