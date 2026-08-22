#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_pipeline.h>   // __pipeline_memcpy_async / __pipeline_commit / __pipeline_wait_prior
                             // （sm_80+ 上映射为 cp.async，global→shared 直达，不过寄存器）
#include <math.h>

// ============================================================
// v1.1 = v1 + K/V 异步搬运（cp.async） + K/V 双缓冲
// ============================================================
// 结构不变：KV 外层、Q 内层、grid=(H,B)、block=Br、一线程一行。
// 只动"搬运"这一件事：
//
//   v1 的问题：加载 K/V → __syncthreads() → 计算，是串行的。
//              搬运的几百 cycle HBM 延迟里，计算单元完全闲置。
//
//   v1.1 做法：给 K/V 各开两块 buffer。
//     算第 j 块的同时，cp.async 在后台预取第 j+1 块；
//     下一轮循环一到，K/V 已经（大概率）在 shared 里了。
//     HBM 延迟被藏进上一块的计算里。
//
//   时间线对比（Tc 个外层块）：
//     v1  : [搬K0][算0][搬K1][算1][搬K2][算2]...
//     v1.1: [搬K0][算0 | 后台搬K1][算1 | 后台搬K2][算2]...
//                └── 延迟与计算重叠 ──┘
//
// shared memory 改为动态分配：
//   双缓冲后 K/V 占 2 倍，Br=32/Bc=32/D=64 时共 49408B，超 48KB 静态上限，
//   必须 extern __shared__ + cudaFuncSetAttribute 解锁（launch 代码里做）。
//   布局（连续一维）：
//     Ks[2][Bc][D] | Vs[2][Bc][D] | Qs[Br][D+1] | Os[Br][D+1]
//   Qs/Os 保留 +1 padding（消 bank conflict）；Ks/Vs 是 broadcast 访问不 padding。
// ============================================================

// ------------------------------------------------------------
// 异步搬运一个 K/V 块到 Ks/Vs 的第 buf 个 buffer
//   - 按 16B（float4）粒度协作铺满，一条 cp.async 搬 4 个 float
//   - 越界行（kv_start+row >= N）用 zfill 全零填充
//     （配合 kernel 里把越界 key 的 S 置 -inf，保证不产生幻影权重）
//   - 对齐要求：src/dst 都 16B 对齐 —— D 是 4 的倍数即满足
// ------------------------------------------------------------
template<int Bc, int HEAD_DIM>
__device__ __forceinline__ void kv_async_copy(
    const float* k, const float* v,   // [B,H,N,D] 全局
    float* Ks, float* Vs,             // shared：各含 2 个 buffer
    size_t bh, int kv_start, int N, int buf)
{
    int tx = threadIdx.x;

    const float4* kg = reinterpret_cast<const float4*>(
        k + bh * N * HEAD_DIM + (size_t)kv_start * HEAD_DIM);
    const float4* vg = reinterpret_cast<const float4*>(
        v + bh * N * HEAD_DIM + (size_t)kv_start * HEAD_DIM);
    float4* ks = reinterpret_cast<float4*>(Ks + (size_t)buf * Bc * HEAD_DIM);
    float4* vs = reinterpret_cast<float4*>(Vs + (size_t)buf * Bc * HEAD_DIM);

    constexpr int NVEC = Bc * HEAD_DIM / 4;      // 块内 float4 总数
    for (int i = tx; i < NVEC; i += blockDim.x) {
        int  row = (i * 4) / HEAD_DIM;           // 该 float4 属于块内第几行
        bool ok  = (kv_start + row) < N;         // 全局行号是否越界
        // 第 4 个参数 zfill：>0 表示末尾这么多字节不读源、直接填 0
        __pipeline_memcpy_async(&ks[i], &kg[i], 16, ok ? 0 : 16);
        __pipeline_memcpy_async(&vs[i], &vg[i], 16, ok ? 0 : 16);
    }
}

template<int Br, int Bc, int HEAD_DIM>
__global__ void __launch_bounds__(Br) flashattention_v1_1(
    const float* q,    // [B, H, N, D]
    const float* k,    // [B, H, N, D]
    const float* v,    // [B, H, N, D]
    float* o,          // [B, H, N, D]
    float* l,          // [B, H, N]  host 初始化 0
    float* m,          // [B, H, N]  host 初始化 -inf
    int B, int H, int N)
{
    static_assert(HEAD_DIM % 4 == 0, "float4 异步搬运要求 D 是 4 的倍数");

    int b  = blockIdx.y;
    int h  = blockIdx.x;
    int tx = threadIdx.x;
    size_t bh = (size_t)b * H + h;

    const float softmax_scale = 1.0f / sqrtf((float)HEAD_DIM);
    const int Tc = (N + Bc - 1) / Bc;

    // ---- 动态 shared memory 布局 ----
    extern __shared__ float smem[];
    float* Ks = smem;                                                        // [2][Bc][D]
    float* Vs = Ks + 2 * Bc * HEAD_DIM;                                      // [2][Bc][D]
    float (*Qs)[HEAD_DIM + 1] =
        reinterpret_cast<float (*)[HEAD_DIM + 1]>(Vs + 2 * Bc * HEAD_DIM);   // [Br][D+1]
    float (*Os)[HEAD_DIM + 1] =
        reinterpret_cast<float (*)[HEAD_DIM + 1]>(&Qs[Br][0]);               // [Br][D+1]

    // ---- prologue：预取第 0 个 K/V 块到 buffer 0 ----
    kv_async_copy<Bc, HEAD_DIM>(k, v, Ks, Vs, bh, 0, N, 0);
    __pipeline_commit();          // 收进"第 0 组"

    // ---- 外层循环：遍历 K/V 块 ----
    for (int j = 0; j < Tc; j++) {
        int kv_start = j * Bc;
        int cur = j & 1;          // 本轮使用的 buffer 编号（j 偶→0，奇→1）

        // 1. 预取第 j+1 块到另一个 buffer（不阻塞，立刻返回）
        if (j + 1 < Tc) {
            kv_async_copy<Bc, HEAD_DIM>(k, v, Ks, Vs, bh, kv_start + Bc, N, cur ^ 1);
            __pipeline_commit();  // 收进"第 j+1 组"
        }

        // 2. 等当前块（第 j 组）到货
        //    wait_prior(1)：允许最新的 1 组还在飞（就是刚发出的 j+1 预取），
        //    其余全部完成 → 第 j 组已就绪。最后一轮没有新预取，必须 wait_prior(0)。
        __pipeline_wait_prior(j + 1 < Tc ? 1 : 0);
        __syncthreads();          // cp.async 只保证"发起线程"可见，同步后全 block 可见

        const float* Kj = Ks + (size_t)cur * Bc * HEAD_DIM;   // 本轮 K 块（flat 视角）
        const float* Vj = Vs + (size_t)cur * Bc * HEAD_DIM;   // 本轮 V 块

        // ---- 内层循环：遍历 Q 块（与 v1 相同）----
        for (int q_start = 0; q_start < N; q_start += Br) {

            // 3. 加载 Q_i（v1.1 保持普通同步搬运；Q 也可异步化，留作后续优化）
            for (int idx = tx; idx < Br * HEAD_DIM; idx += blockDim.x) {
                int row  = idx / HEAD_DIM;
                int col  = idx % HEAD_DIM;
                int grow = q_start + row;
                if (grow < N) {
                    Qs[row][col] = q[bh * N * HEAD_DIM + grow * HEAD_DIM + col];
                }
            }
            __syncthreads();

            // 4. 读回 O_i / l_i / m_i
            int row  = tx;
            int grow = q_start + row;
            float m_reg;
            float l_reg;

            if (grow < N) {
                m_reg = m[bh * N + grow];
                l_reg = l[bh * N + grow];
                #pragma unroll
                for (int d = 0; d < HEAD_DIM; d++)
                    Os[row][d] = o[bh * N * HEAD_DIM + grow * HEAD_DIM + d];
            }

            // 5. 计算 S = Qs @ Kj^T
            //    与 v1 的差别只有两处：
            //    a) Ks 改为 flat 指针 Kj（同一个 buffer 里的 [Bc][D] 行主序，访存模式不变）
            //    b) 越界 key 的分数置 -inf（K 越界行已被 zfill 清零，这里挡住幻影权重；
            //       N 是 Bc 整数倍时永不触发，零开销）
            float S[Bc];

            #pragma unroll
            for (int c = 0; c < Bc; c++) {
                const float* krow = Kj + c * HEAD_DIM;
                float s = 0.0f;
                #pragma unroll
                for (int d = 0; d < HEAD_DIM; d++)
                    s += Qs[row][d] * krow[d];
                S[c] = (kv_start + c < N) ? s * softmax_scale : -INFINITY;
            }

            // 6. online softmax 更新（与 v1 完全相同）
            float m_tilde = -INFINITY;
            #pragma unroll
            for (int c = 0; c < Bc; c++)
                m_tilde = fmaxf(m_tilde, S[c]);

            float l_tilde = 0.0f;
            #pragma unroll
            for (int c = 0; c < Bc; c++)
                l_tilde += __expf(S[c] - m_tilde);      // S[c]=-inf 时贡献 0

            float m_new = fmaxf(m_reg, m_tilde);
            float l_new = __expf(m_reg - m_new) * l_reg + __expf(m_tilde - m_new) * l_tilde;

            float scale_old = __expf(m_reg - m_new) * l_reg;
            float scale_new = __expf(m_tilde - m_new);

            #pragma unroll
            for (int d = 0; d < HEAD_DIM; d++) {
                float pv = 0.0f;
                #pragma unroll
                for (int c = 0; c < Bc; c++)
                    pv += __expf(S[c] - m_tilde) * Vj[c * HEAD_DIM + d];
                Os[row][d] = (scale_old * Os[row][d] + scale_new * pv) / l_new;
            }

            // 7. 写回 O_i / l_i / m_i
            if (grow < N) {
                m[bh * N + grow] = m_new;
                l[bh * N + grow] = l_new;
                #pragma unroll
                for (int d = 0; d < HEAD_DIM; d++)
                    o[bh * N * HEAD_DIM + grow * HEAD_DIM + d] = Os[row][d];
            }

            // 8. 内层收尾同步：
            //    a) 保护 Qs/Os 下一轮覆盖写
            //    b) 保证所有线程读完 Kj/Vj —— 下一轮外层顶部的预取（第 j+2 块）
            //       会写回 buffer cur，没有这个同步就会踩还没读完的数据
            __syncthreads();
        }
    }
}
