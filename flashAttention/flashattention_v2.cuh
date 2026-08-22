#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_pipeline.h>   // cp.async（__pipeline_*）
#include <math.h>

// ============================================================
// FlashAttention v2.0（标量版，CUDA Core）
// ============================================================
// 相对 v1/v1.1 的结构改造（对应 v2 论文三大改进的标量形态）：
//
//  1. 循环换边：Q 外层、K/V 内层
//     grid = (ceil(N/Br), B*H)   ← Q 块并行到 grid，block 数随 N 增长
//     block = Br 线程，一线程一行
//     → Br=128 时 block=4 warp，每 warp 独占 32 行、互不通信：
//       这就是 v2 论文"warp 沿 Q 切分"的标量形态
//
//  2. m/l/O 状态全程片上：
//     Q 只加载一次，驻留 shared 整个内循环（官方注释 "stay in SRAM throughout"）
//     m/l 是每线程 2 个寄存器，kernel 内初始化，不再有 HBM 的 m/l 数组
//     → 删掉了 init_state_kernel，O 的 HBM 往返从 2N²D/Bc 降到 2ND
//
//  3. 延迟归一化：维护「未归一化」累加器 Õ
//     内循环:  Õ = e^{m_old−m_new}·Õ + e^{m̃−m_new}·(P̃@V)   ← 只有乘加，无除法
//     结束后:  O = Õ / l                                      ← 只除一次
//     对比 v1 每轮 (…)/l_new：Tc 轮省 Tc×D 次除法
//
//  4. 保留 v1.1 的 K/V cp.async 双缓冲（内循环恰好流式过 K/V，逻辑原样平移）
//     新增：P[c] 预存，PV 内积不再重复计算 exp
//
//  5. Causal mask 优化（官方 Triton 版思路，两级粒度）：
//     a) 块级跳过 —— 不进循环。第 q 行只能看 key 编号 <= q 的位置，
//        所以本 Q 块（行号 [q_start, q_start+Br)）根本不用碰
//        起点在 q_start+Br 之后的任何 K/V 块。
//        实现：把内循环的结束位置从 N 砍到 min(q_start+Br, N)，
//        对角线以上的块（约一半计算量）直接消失，一行 if 都不用
//     b) 逐元素 mask —— 只给「跨对角线的块」加。整块可见的块
//        （块内最大 key 编号 <= q_start）完全不用 mask；
//        只有斜跨对角线的那一块需要逐元素判断：
//        key 编号 kv_start+c > 本线程行号 grow → S[c] = -inf
//        （-inf 过 exp 后变 0，softmax 权重自然为 0）
//
// shared 布局（动态）：Ks[2][Bc][D] | Vs[2][Bc][D] | Qs[Br][D+1] | Oacc[Br][D+1]
//   Qs/Oacc 加 +1 padding（一线程一行，按行访问，消 bank conflict）
//   Br=128/Bc=32/D=64 时共 99328B ≈ 97KB，需 cudaFuncSetAttribute 解锁（bench 里做）
// ============================================================

// K/V 块异步搬运（与 v1.1 相同）：16B 粒度铺满，越界行 zfill 清零
template<int Bc, int HEAD_DIM>
__device__ __forceinline__ void kv_async_copy_v2(
    const float* k, const float* v,
    float* Ks, float* Vs,
    size_t bh, int kv_start, int N, int buf)
{
    int tx = threadIdx.x;

    const float4* kg = reinterpret_cast<const float4*>(
        k + bh * N * HEAD_DIM + (size_t)kv_start * HEAD_DIM);
    const float4* vg = reinterpret_cast<const float4*>(
        v + bh * N * HEAD_DIM + (size_t)kv_start * HEAD_DIM);
    float4* ks = reinterpret_cast<float4*>(Ks + (size_t)buf * Bc * HEAD_DIM);
    float4* vs = reinterpret_cast<float4*>(Vs + (size_t)buf * Bc * HEAD_DIM);

    constexpr int NVEC = Bc * HEAD_DIM / 4;
    for (int i = tx; i < NVEC; i += blockDim.x) {
        int  row = (i * 4) / HEAD_DIM;
        bool ok  = (kv_start + row) < N;
        __pipeline_memcpy_async(&ks[i], &kg[i], 16, ok ? 0 : 16);
        __pipeline_memcpy_async(&vs[i], &vg[i], 16, ok ? 0 : 16);
    }
}

template<bool CAUSAL, int Br, int Bc, int HEAD_DIM>
__global__ void __launch_bounds__(Br) flashattention_v2(
    const float* q,    // [B, H, N, D]
    const float* k,    // [B, H, N, D]
    const float* v,    // [B, H, N, D]
    float* o,          // [B, H, N, D]  只写一次
    int N)             // 序列长度（B*H 已合并进 gridDim.y）
{
    static_assert(HEAD_DIM % 4 == 0, "float4 异步搬运要求 D 是 4 的倍数");

    int qb = blockIdx.x;           // Q 块编号 [0, ceil(N/Br))
    int bh = blockIdx.y;           // batch*head 合并索引
    int tx = threadIdx.x;

    int  q_start  = qb * Br;       // 本 block 负责的 Q 行起点
    int  row      = tx;            // 块内行号（一线程一行）
    int  grow     = q_start + row; // 全局 query 行号
    bool row_valid = grow < N;     // 末块可能越界，越界线程不写回

    const float softmax_scale = 1.0f / sqrtf((float)HEAD_DIM);

    // ---- causal 块级跳过：裁剪内循环的结束位置 ----
    // 第 grow 行能看到的 key 编号最大是 grow，本 Q 块里最大的行号是 q_start+Br-1，
    // 所以最后一个可能可见的 key 编号是 q_start+Br-1，再 +1 得「开区间上界」kv_limit。
    // 非 causal 时上界就是 N。块数 Tc 由上界算出，对角线以上的块根本不进循环。
    // 例：N=2048、Br=128、qb=5 时本块行号 [640,768)，只处理 key < 768 的块，
    //     768 之后的 128 个块（N/Bc=64 个中的 40 个）直接跳过。
    const int kv_limit = CAUSAL ? min(q_start + Br, N) : N;
    const int Tc = (kv_limit + Bc - 1) / Bc;   // 本 Q 块实际要处理的 K/V 块数

    const size_t base = (size_t)bh * N * HEAD_DIM;

    // ---- 动态 shared 布局 ----
    extern __shared__ float smem[];
    float* Ks   = smem;                                                      // [2][Bc][D]
    float* Vs   = Ks + 2 * Bc * HEAD_DIM;                                    // [2][Bc][D]
    float (*Qs)[HEAD_DIM + 1] =
        reinterpret_cast<float (*)[HEAD_DIM + 1]>(Vs + 2 * Bc * HEAD_DIM);   // [Br][D+1]
    float (*Oacc)[HEAD_DIM + 1] =
        reinterpret_cast<float (*)[HEAD_DIM + 1]>(&Qs[Br][0]);               // [Br][D+1] 未归一化 Õ

    // ---- Q 加载一次，驻留整个内循环 ----
    for (int idx = tx; idx < Br * HEAD_DIM; idx += blockDim.x) {
        int r = idx / HEAD_DIM, c = idx % HEAD_DIM;
        if (q_start + r < N)
            Qs[r][c] = q[base + (size_t)(q_start + r) * HEAD_DIM + c];
    }

    // ---- 每线程自己的状态：m/l 进寄存器（kernel 内初始化，无 HBM 往返）----
    float m_reg = -INFINITY;
    float l_reg = 0.0f;
    #pragma unroll
    for (int d = 0; d < HEAD_DIM; d++)
        Oacc[row][d] = 0.0f;      // Õ 清零

    __syncthreads();

    // ---- prologue：预取第 0 个 K/V 块 ----
    kv_async_copy_v2<Bc, HEAD_DIM>(k, v, Ks, Vs, bh, 0, N, 0);
    __pipeline_commit();

    // ================= 内层循环：流式过所有 K/V 块 =================
    for (int j = 0; j < Tc; j++) {
        int kv_start = j * Bc;
        int cur = j & 1;

        // 预取 j+1 块到另一个 buffer（后台飞）
        if (j + 1 < Tc) {
            kv_async_copy_v2<Bc, HEAD_DIM>(k, v, Ks, Vs, bh, kv_start + Bc, N, cur ^ 1);
            __pipeline_commit();
        }
        __pipeline_wait_prior(j + 1 < Tc ? 1 : 0);
        __syncthreads();

        const float* Kj = Ks + (size_t)cur * Bc * HEAD_DIM;
        const float* Vj = Vs + (size_t)cur * Bc * HEAD_DIM;

        // ---- causal 逐元素 mask 的块分类（在 S 计算前判断一次，循环里复用）----
        // 整块可见：本 K/V 块的最大 key 编号 kv_start+Bc-1 <= 本 Q 块的最小行号 q_start
        //           → 本块所有行都看得见所有 key，不需要任何 mask（大多数块属于此类）
        // 跨对角线：kv_start+Bc-1 > q_start 且本块还在循环里
        //           → 块内有部分 key 超过了块内靠前的行，需要逐元素判断
        const bool diag_block = CAUSAL && (kv_start + Bc - 1 > q_start);

        // 1. S = Qs[row] · Kj^T（越界 key 置 -inf，杜绝幻影权重）
        float S[Bc];
        #pragma unroll
        for (int c = 0; c < Bc; c++) {
            const float* krow = Kj + c * HEAD_DIM;
            float s = 0.0f;
            #pragma unroll
            for (int d = 0; d < HEAD_DIM; d++)
                s += Qs[row][d] * krow[d];
            S[c] = (kv_start + c < N) ? s * softmax_scale : -INFINITY;

            // causal：key 编号超过「我这个线程的行号 grow」就不可见（下三角）
            // 只在跨对角线的块里判，注意是 grow（行级）不是 q_start（块级）
            if (diag_block && kv_start + c > grow)
                S[c] = -INFINITY;
        }

        // 2. 本块统计：m̃ / P̃ / l̃（P 预存，PV 内积不再重复算 exp）
        float m_tilde = -INFINITY;
        #pragma unroll
        for (int c = 0; c < Bc; c++)
            m_tilde = fmaxf(m_tilde, S[c]);

        float P[Bc];
        float l_tilde = 0.0f;
        #pragma unroll
        for (int c = 0; c < Bc; c++) {
            P[c] = __expf(S[c] - m_tilde);    // S[c]=-inf 时 P[c]=0
            l_tilde += P[c];
        }

        // 3. 新旧合并（汇率换算到新基准 m_new）
        float m_new     = fmaxf(m_reg, m_tilde);
        float scale_old = __expf(m_reg - m_new);    // 历史 Õ 换基准
        float scale_new = __expf(m_tilde - m_new);  // 本块 P̃V 换基准
        l_reg = scale_old * l_reg + scale_new * l_tilde;
        m_reg = m_new;

        // 4. Õ 累加（延迟归一化：不除 l，纯乘加）
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; d++) {
            float pv = 0.0f;
            #pragma unroll
            for (int c = 0; c < Bc; c++)
                pv += P[c] * Vj[c * HEAD_DIM + d];
            Oacc[row][d] = scale_old * Oacc[row][d] + scale_new * pv;
        }

        // 末尾同步：保证所有线程读完 cur buffer，下一轮预取 j+2 才能覆盖它
        __syncthreads();
    }

    // ---- 最后一次归一化 + 写回（O 的唯一一次 HBM 写）----
    if (row_valid) {
        float inv_l = 1.0f / l_reg;    // 倒数，D 次除法变 1 次除法 + D 次乘法
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; d++)
            o[base + (size_t)grow * HEAD_DIM + d] = Oacc[row][d] * inv_l;
    }
}
