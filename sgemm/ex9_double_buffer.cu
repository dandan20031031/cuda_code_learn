#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

#define BM  128
#define BN  128
#define BK  8

#define vload(v, addr)   v = *((float4*)(addr))

// 列主序共享内存: sa[col*BM+row], sb[k*BN+n]
__global__ __launch_bounds__(256)
void sgemm_k11(float* A, float* B, float* C, int M, int N, int K)
{
    int tx = threadIdx.x;
    int bx = blockIdx.x, by = blockIdx.y;

    int warp_id = tx >> 5,       lane_id = tx & 31;
    int warp_row = warp_id & 3,  warp_col = warp_id >> 2;
    int row_w = lane_id & 3,     col_w = lane_id >> 2;

    int row_c = (warp_row << 5) + (row_w << 3);   // 0..127
    int col_c = (warp_col << 6) + (col_w << 3);   // 0..127

    int a_row = tx >> 1,         a_col = (tx & 1) << 2;  // A 加载
    int b_row = (tx & 1) << 2,   b_col = tx >> 1;        // B 加载

    // 双缓冲共享内存 (16KB)
    __shared__ float sa[2][BM * BK];   // 2 × 1KB
    __shared__ float sb[2][BN * BK];   // 2 × 1KB

    float4 Av1[2], Av2[2], Bv1[2], Bv2[2], Cres[16];
    float4 pref_A, pref_B;
    memset(Cres, 0, sizeof(Cres));

    int K_tiles = K >> 3;
    float* A_base = A + by * BM * K;
    float* B_base = B + bx * BN;

    // ============================================================
    // Prologue: 加载 tile 0 → buffer[0]
    // ============================================================
    {
        float* ptr_A = A_base;
        float* ptr_B = B_base;
        vload(pref_A, &ptr_A[a_row * K + a_col]);
        int bb = b_row * N + b_col;
        pref_B.x = ptr_B[bb];
        pref_B.y = ptr_B[bb + N];
        pref_B.z = ptr_B[bb + 2*N];
        pref_B.w = ptr_B[bb + 3*N];

        // 写入 buffer[0]
        float* psa = (float*)sa;   // &sa[0][0]
        float* psb = (float*)sb;
        psa[a_col * BM + a_row] = pref_A.x;
        psa[(a_col+1) * BM + a_row] = pref_A.y;
        psa[(a_col+2) * BM + a_row] = pref_A.z;
        psa[(a_col+3) * BM + a_row] = pref_A.w;
        psb[b_row * BN + b_col] = pref_B.x;
        psb[(b_row+1) * BN + b_col] = pref_B.y;
        psb[(b_row+2) * BN + b_col] = pref_B.z;
        psb[(b_row+3) * BN + b_col] = pref_B.w;
    }
    __syncthreads();

    // 预取 buffer[0] 的 inner_k=0
    {
        float* psa = (float*)sa;
        float* psb = (float*)sb;
        vload(Av1[0], &psa[0 * BM + row_c]);
        vload(Av2[0], &psa[0 * BM + row_c + 4]);
        vload(Bv1[0], &psb[0 * BN + col_c]);
        vload(Bv2[0], &psb[0 * BN + col_c + 4]);
    }

    // ============================================================
    // Main Loop: 双缓冲
    // ============================================================
    for (int kc = 0; kc < K_tiles; kc++) {
        // ① LDG: 预取 tile (kc+1) → 寄存器
        int inc = (kc + 1) % K_tiles;
        if (kc + 1 < K_tiles) {
            float* next_A = A_base + inc * BK;
            float* next_B = B_base + inc * BK * N;
            vload(pref_A, &next_A[a_row * K + a_col]);
            int bb = b_row * N + b_col;
            pref_B.x = next_B[bb];
            pref_B.y = next_B[bb + N];
            pref_B.z = next_B[bb + 2*N];
            pref_B.w = next_B[bb + 3*N];
        }

        // ② 指向当前 buffer (kc 偶→buf0, 奇→buf1)
        int cur_buf = kc & 1;
        float* psa = (float*)sa + cur_buf * BM * BK;
        float* psb = (float*)sb + cur_buf * BN * BK;

        // ③ 内层循环: 寄存器乒乓, 读当前 buffer
        #pragma unroll
        for (int ik = 0; ik < BK; ik++) {
            int next_ik = (ik + 1) & 7;
            vload(Av1[(ik + 1) & 1], &psa[next_ik * BM + row_c]);
            vload(Av2[(ik + 1) & 1], &psa[next_ik * BM + row_c + 4]);
            vload(Bv1[(ik + 1) & 1], &psb[next_ik * BN + col_c]);
            vload(Bv2[(ik + 1) & 1], &psb[next_ik * BN + col_c + 4]);

            float4 a1 = Av1[ik & 1], a2 = Av2[ik & 1];
            float4 b1 = Bv1[ik & 1], b2 = Bv2[ik & 1];
            #define FMA(c,a,b) c.x+=a.x*b; c.y+=a.y*b; c.z+=a.z*b; c.w+=a.w*b;
            FMA(Cres[0],a1,b1.x); FMA(Cres[1],a2,b1.x);
            FMA(Cres[2],a1,b1.y); FMA(Cres[3],a2,b1.y);
            FMA(Cres[4],a1,b1.z); FMA(Cres[5],a2,b1.z);
            FMA(Cres[6],a1,b1.w); FMA(Cres[7],a2,b1.w);
            FMA(Cres[8],a1,b2.x); FMA(Cres[9],a2,b2.x);
            FMA(Cres[10],a1,b2.y); FMA(Cres[11],a2,b2.y);
            FMA(Cres[12],a1,b2.z); FMA(Cres[13],a2,b2.z);
            FMA(Cres[14],a1,b2.w); FMA(Cres[15],a2,b2.w);
            #undef FMA
        }

        // ④ 写入下一 tile 到另一 buffer (无需 __syncthreads 因为读/写不同 buffer)
        if (kc + 1 < K_tiles) {
            int nxt_buf = (kc + 1) & 1;
            float* psa_nxt = (float*)sa + nxt_buf * BM * BK;
            float* psb_nxt = (float*)sb + nxt_buf * BN * BK;
            psa_nxt[a_col * BM + a_row] = pref_A.x;
            psa_nxt[(a_col+1) * BM + a_row] = pref_A.y;
            psa_nxt[(a_col+2) * BM + a_row] = pref_A.z;
            psa_nxt[(a_col+3) * BM + a_row] = pref_A.w;
            psb_nxt[b_row * BN + b_col] = pref_B.x;
            psb_nxt[(b_row+1) * BN + b_col] = pref_B.y;
            psb_nxt[(b_row+2) * BN + b_col] = pref_B.z;
            psb_nxt[(b_row+3) * BN + b_col] = pref_B.w;
            __syncthreads();
        }

        // ⑤ 从新 buffer 预取 inner_k=0 (下次迭代用)
        {
            int nxt_buf = (kc + 1) & 1;
            float* psa2 = (float*)sa + nxt_buf * BM * BK;
            float* psb2 = (float*)sb + nxt_buf * BN * BK;
            vload(Av1[0], &psa2[0 * BM + row_c]);
            vload(Av2[0], &psa2[0 * BM + row_c + 4]);
            vload(Bv1[0], &psb2[0 * BN + col_c]);
            vload(Bv2[0], &psb2[0 * BN + col_c + 4]);
        }
    }

    // ============================================================
    // Write Back
    // ============================================================
    float* C_block = C + by * BM * N + bx * BN;
    #define W(x, ro, co) C_block[(row_c + ro) * N + col_c + co] = x
    W(Cres[0].x,0,0); W(Cres[0].y,1,0); W(Cres[0].z,2,0); W(Cres[0].w,3,0);
    W(Cres[1].x,4,0); W(Cres[1].y,5,0); W(Cres[1].z,6,0); W(Cres[1].w,7,0);
    W(Cres[2].x,0,1); W(Cres[2].y,1,1); W(Cres[2].z,2,1); W(Cres[2].w,3,1);
    W(Cres[3].x,4,1); W(Cres[3].y,5,1); W(Cres[3].z,6,1); W(Cres[3].w,7,1);
    W(Cres[4].x,0,2); W(Cres[4].y,1,2); W(Cres[4].z,2,2); W(Cres[4].w,3,2);
    W(Cres[5].x,4,2); W(Cres[5].y,5,2); W(Cres[5].z,6,2); W(Cres[5].w,7,2);
    W(Cres[6].x,0,3); W(Cres[6].y,1,3); W(Cres[6].z,2,3); W(Cres[6].w,3,3);
    W(Cres[7].x,4,3); W(Cres[7].y,5,3); W(Cres[7].z,6,3); W(Cres[7].w,7,3);
    W(Cres[8].x,0,4); W(Cres[8].y,1,4); W(Cres[8].z,2,4); W(Cres[8].w,3,4);
    W(Cres[9].x,4,4); W(Cres[9].y,5,4); W(Cres[9].z,6,4); W(Cres[9].w,7,4);
    W(Cres[10].x,0,5); W(Cres[10].y,1,5); W(Cres[10].z,2,5); W(Cres[10].w,3,5);
    W(Cres[11].x,4,5); W(Cres[11].y,5,5); W(Cres[11].z,6,5); W(Cres[11].w,7,5);
    W(Cres[12].x,0,6); W(Cres[12].y,1,6); W(Cres[12].z,2,6); W(Cres[12].w,3,6);
    W(Cres[13].x,4,6); W(Cres[13].y,5,6); W(Cres[13].z,6,6); W(Cres[13].w,7,6);
    W(Cres[14].x,0,7); W(Cres[14].y,1,7); W(Cres[14].z,2,7); W(Cres[14].w,3,7);
    W(Cres[15].x,4,7); W(Cres[15].y,5,7); W(Cres[15].z,6,7); W(Cres[15].w,7,7);
    #undef W
}

// ============================================================
#define CUDA_CHK(call) do { cudaError_t _e_=(call); \
    if(_e_!=cudaSuccess){ printf("CUDA ERROR: %s\n",cudaGetErrorString(_e_)); exit(1); } \
} while(0)

float timeKernel(void (*fn)(float*,float*,float*,int,int,int),
                 dim3 g,dim3 b,float* dA,float* dB,float* dC,
                 int M,int N,int K,const char* label){
    cudaEvent_t s,e; cudaEventCreate(&s); cudaEventCreate(&e);
    fn<<<g,b>>>(dA,dB,dC,M,N,K); CUDA_CHK(cudaDeviceSynchronize());
    cudaEventRecord(s);
    fn<<<g,b>>>(dA,dB,dC,M,N,K); CUDA_CHK(cudaGetLastError());
    cudaEventRecord(e); CUDA_CHK(cudaEventSynchronize(e));
    float ms; cudaEventElapsedTime(&ms,s,e);
    printf("  %-30s  %.4f ms\n",label,ms);
    cudaEventDestroy(s); cudaEventDestroy(e); return ms;
}

void sgemm_cpu(float* A,float* B,float* C,int M,int N,int K){
    memset(C,0,M*N*sizeof(float));
    for(int i=0;i<M;i++) for(int j=0;j<N;j++)
        for(int k=0;k<K;k++) C[i*N+j]+=A[i*K+k]*B[k*N+j];
}
void init(float* m,int n){ for(int i=0;i<n;i++) m[i]=(rand()%100)/100.0f; }

int main(){
    srand(42);
    int M=4096,N=4096,K=4096;
    size_t szA=M*K*sizeof(float),szB=K*N*sizeof(float),szC=M*N*sizeof(float);

    printf("\n=== ex9: Double Buffering ===\n");
    printf("  BM=%d BN=%d BK=%d  Shared=16KB  Block=256  Grid=%dx%d\n",
           BM,BN,BK,(N+BN-1)/BN,(M+BM-1)/BM);

    float *hA=(float*)malloc(szA),*hB=(float*)malloc(szB);
    float *hC_cpu=(float*)malloc(szC),*hC_gpu=(float*)malloc(szC);
    init(hA,M*K); init(hB,K*N);

    float *dA,*dB,*dC;
    CUDA_CHK(cudaMalloc(&dA,szA)); CUDA_CHK(cudaMalloc(&dB,szB));
    CUDA_CHK(cudaMalloc(&dC,szC));
    CUDA_CHK(cudaMemcpy(dA,hA,szA,cudaMemcpyHostToDevice));
    CUDA_CHK(cudaMemcpy(dB,hB,szB,cudaMemcpyHostToDevice));

    dim3 block(256);
    dim3 grid((N+BN-1)/BN,(M+BM-1)/BM);

    float ms=timeKernel(sgemm_k11,grid,block,dA,dB,dC,M,N,K,"Kernel 11: double buf");

    CUDA_CHK(cudaMemcpy(hC_gpu,dC,szC,cudaMemcpyDeviceToHost));
    sgemm_cpu(hA,hB,hC_cpu,M,N,K);
    float maxDiff=0,sumCPU=0;
    for(int i=0;i<M*N;i++){
        float d=fabsf(hC_gpu[i]-hC_cpu[i]);
        if(d>maxDiff) maxDiff=d; sumCPU+=fabsf(hC_cpu[i]);
    }
    float tol=sumCPU/(M*N)*1e-3f;
    printf("  Max diff: %e (tol=%.2e) → %s\n",maxDiff,tol,maxDiff<tol?"PASS":"FAIL");
    double gflops=2.0*M*N*K/(ms*1e6);
    printf("  Performance: %.1f GFLOPS\n",gflops);

    CUDA_CHK(cudaFree(dA)); CUDA_CHK(cudaFree(dB)); CUDA_CHK(cudaFree(dC));
    free(hA); free(hB); free(hC_cpu); free(hC_gpu);
    return 0;
}
