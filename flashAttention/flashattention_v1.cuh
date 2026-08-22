#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// ============================================================
// grid / block 划分
// ============================================================
// grid = (H, B)：
//   blockIdx.x = h   （注意力头 head）
//   blockIdx.y = b   （batch）
// 每个 thread block 负责一个 (b, h) 的完整 attention（N×N）
// block 内部：外层循环遍历 K/V 块（步长 Bc），内层循环遍历 Q 块（步长 Br）
// ============================================================

// ============================================================
// Q/K/V/O 张量形状与索引（重点理解这里）
// ============================================================
// 形状统一为 [B, H, N, D]，连续内存（row-major 展平）：
//   B = batch     批次数
//   H = head      注意力头数
//   N = seq_len   序列长度（token 个数）
//   D = head_dim  每个头的特征维度（本实现 = HEAD_DIM）
//
// 元素 (b, h, n, d) 在内存中的偏移：
//   offset = ((b * H + h) * N + n) * D + d
//
// 逐维拆解（每维"跨一步"的大小）：
//   b 维：H * N * D     ← 跳到第 b 个 batch
//   h 维：N * D         ← 跳到第 h 个 head
//   n 维：D             ← 跳到第 n 个 token（序列位置）
//   d 维：1             ← 第 d 维特征
//
// 固定 (b, h) 后，其切片是 [N, D] 的矩阵：
//   第 n 行 = 第 n 个 token，第 d 列 = 第 d 维特征
//   偏移 = (b*H + h) * N*D + n*D + d
//
// 例：读 K 的第 (b,h,n,d) 个元素：
//   k[ ((b*H + h) * N + n) * D + d ]
// ============================================================

template<int Br, int Bc, int HEAD_DIM>
__global__ void flashattention_v1(
    const float* q,    // [B, H, N, D]
    const float* k,    // [B, H, N, D]
    const float* v,    // [B, H, N, D]
    float* o,          // [B, H, N, D]  输出（论文里的 O，走 HBM）
    float* l,          // [B, H, N]     softmax 分母（指数和），HBM 持久化，host 初始化 0
    float* m,          // [B, H, N]     softmax 运行最大值，HBM 持久化，host 初始化 -inf
    int B, int H, int N)
{
    // ---- 当前 block 负责的 (batch, head) ----
    int b  = blockIdx.y;               // batch
    int h  = blockIdx.x;               // head
    int tx = threadIdx.x;
    size_t bh = (size_t)b * H + h;     // batch*head 合并偏移（进入第 bh 个 [N,D] 切片）

    const float softmax_scale = 1.0f / sqrtf((float)HEAD_DIM);   // HEAD_DIM 是编译期常量，编译器会折叠为常量

    __shared__ float Qs[Br][HEAD_DIM + 1];   // Q 块：Br 行 × HEAD_DIM 列（+1 padding 消除 bank conflict）
    __shared__ float Ks[Bc][HEAD_DIM];       // K 块：Bc 行 × HEAD_DIM 列（broadcast 访问，无需 padding）
    __shared__ float Vs[Bc][HEAD_DIM];       // V 块：Bc 行 × HEAD_DIM 列（broadcast 访问，无需 padding）
    __shared__ float Os[Br][HEAD_DIM + 1];   // O 块：Br 行 × HEAD_DIM 列（+1 padding 消除 bank conflict）

    // ---- 外层循环：遍历 K/V 块（步长 Bc）----
    // 论文 Algorithm 1 第 5 行：for j = 1..Tc（外层 K/V）
    for (int kv_start = 0; kv_start < N; kv_start += Bc) {

        // 1. 加载 K_j、V_j（Bc 行 × HEAD_DIM 列）到 shared memory
        //    论文第 6 行：Load K_j, V_j from HBM to SRAM
        for (int idx = tx; idx < Bc * HEAD_DIM; idx += blockDim.x) {
            int row  = idx / HEAD_DIM;   // 块内行号 [0, Bc)
            int col  = idx % HEAD_DIM;   // 列号 [0, HEAD_DIM)
            int grow = kv_start + row;   // 全局序列位置（行号）

            if (grow < N) {
                Ks[row][col] = k[bh * N * HEAD_DIM + grow * HEAD_DIM + col];
                Vs[row][col] = v[bh * N * HEAD_DIM + grow * HEAD_DIM + col];
            }
        }
        __syncthreads();

        // ---- 内层循环：遍历 Q 块（步长 Br）----
        // 论文 Algorithm 1 第 7 行：for i = 1..Tr（内层 Q）
        for (int q_start = 0; q_start < N; q_start += Br) {

            // 2. 加载 Q_i（Br 行 × HEAD_DIM 列）到 shared memory
            for (int idx = tx; idx < Br * HEAD_DIM; idx += blockDim.x) {
                int row  = idx / HEAD_DIM;
                int col  = idx % HEAD_DIM;
                int grow = q_start + row;
                if (grow < N) {
                    Qs[row][col] = q[bh * N * HEAD_DIM + grow * HEAD_DIM + col];
                }
            }
            __syncthreads();

            // 3. 读回 O_i / l_i / m_i（论文第 8 行：Load O_i, l_i, m_i from HBM）
            //    —— 这是「K/V 外层」的关键：m/l/O 是 N 行的全局状态，跨外层迭代持久化。
            //    O_i 就在 o 数组（第 q_start..q_start+Br 行），l_i/m_i 在 l/m 数组。
            //    分工：m/l 是标量放寄存器，O 读进 SRAM 的 Os（每线程一行）。
            //    约束：launch 时需 blockDim.x == Br（每个线程负责 Q 块的一行）。
            int row  = tx;                      // 块内行号 [0, Br) 每个线程负责一行
            int grow = q_start + row;           // 全局序列位置
            float m_reg;                        // 运行最大值（标量，放寄存器）
            float l_reg;                        // 指数和（标量，放寄存器）

            if (grow < N) {
                m_reg = m[bh * N + grow];       // [B,H,N] 标量
                l_reg = l[bh * N + grow];       // [B,H,N] 标量
                #pragma unroll
                for (int d = 0; d < HEAD_DIM; d++)
                    Os[row][d] = o[bh * N * HEAD_DIM + grow * HEAD_DIM + d];  // O 读进 SRAM 的 Os
            }

            // 4. 计算 S = Qs @ Ks^T（每线程只算自己那行，Bc 个分数，放寄存器）
            //    S[c] = sum_d Qs[row][d] * Ks[c][d] * softmax_scale
            float S[Bc];                        // 每线程一行
                                                // 注：只有"所有"访问下标都是编译期常量时 S 才能进寄存器；
                                                // 写循环也必须 unroll，否则 S 落 local memory（LDL/STL）

            #pragma unroll
            for (int c = 0; c < Bc; c++) {
                float s = 0.0f;                 // 用局部变量累加内积
                #pragma unroll
                for (int d = 0; d < HEAD_DIM; d++)
                    s += Qs[row][d] * Ks[c][d];
                S[c] = s * softmax_scale;       // 内积算完再乘 scale
            }

            // 5. online softmax 更新（论文第 10-13 行，归一化形式）
            //    m̃     = rowmax(S[:])                     // 当前块这行的最大分数
            //    P̃[c]  = exp(S[c] - m̃)                    // 减最大再指数（数值稳定）
            //    l̃     = rowsum(P̃[:])                     // 当前块这行的 exp 之和
            //    m_new = max(m_reg, m̃)                    // 和历史最大合并
            //    l_new = e^{m_reg - m_new} * l_reg + e^{m̃ - m_new} * l̃
            //    O_new[:] = (1/l_new) * ( e^{m_reg - m_new} * l_reg * Os[row][:]
            //                              + e^{m̃ - m_new} * (P̃[:] @ Vs) )
            //    注意：论文维护「归一化」的 O（每次迭代都除 l_new 再写回）

            // 5a. 当前块这行的最大分数 m̃ = rowmax(S)
            float m_tilde = -INFINITY;
            #pragma unroll
            for (int c = 0; c < Bc; c++)
                m_tilde = fmaxf(m_tilde, S[c]);

            // 5b. 当前块这行的 exp 之和 l̃ = rowsum(exp(S - m̃))
            float l_tilde = 0.0f;
            #pragma unroll
            for (int c = 0; c < Bc; c++)
                l_tilde += __expf(S[c] - m_tilde);

            // 5c. 和历史状态合并，得到全局最大 m_new 和分母 l_new
            float m_new = fmaxf(m_reg, m_tilde);
            float l_new = __expf(m_reg - m_new) * l_reg + __expf(m_tilde - m_new) * l_tilde;

            // 5d. 更新 O（论文第 15 行，归一化形式）
            //     O_new[d] = (1/l_new) * ( e^{m_reg-m_new} * l_reg * Os[row][d]
            //                               + e^{m_tilde-m_new} * (sum_c P[c] * Vs[c][d]) )
            //     其中 P[c] = exp(S[c] - m_tilde) 是当前块的 softmax 权重
            float scale_old = __expf(m_reg - m_new) * l_reg;   // 历史 O 的缩放系数
            float scale_new = __expf(m_tilde - m_new);          // 当前块 P̃V 的缩放系数

            #pragma unroll
            for (int d = 0; d < HEAD_DIM; d++) {
                float pv = 0.0f;                                // P̃ @ V 的第 d 个元素
                #pragma unroll
                for (int c = 0; c < Bc; c++)
                    pv += __expf(S[c] - m_tilde) * Vs[c][d];    // 注：exp 每个 d 重算，可优化预存 P[c]
                Os[row][d] = (scale_old * Os[row][d] + scale_new * pv) / l_new;
            }

            // 6. 写回 O_i / l_i / m_i（论文第 15/17 行：写回 HBM）
            if (grow < N) {
                m[bh * N + grow] = m_new;
                l[bh * N + grow] = l_new;
                #pragma unroll
                for (int d = 0; d < HEAD_DIM; d++)
                    o[bh * N * HEAD_DIM + grow * HEAD_DIM + d] = Os[row][d];
            }

            __syncthreads();   // 下一轮覆盖写 Qs/Ks/Vs，先等所有线程用完
        }
    }
}
