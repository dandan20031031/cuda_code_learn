#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <math.h>
//一个线程负责计算一行的softmax值
__global__ void softmax_v1_naive(
    float *input,   // M×K, row-major
    float *output,    // M×K, row-major
    int M, int N)
    {
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        if (row >= M) return;
        float *x = input + row * N;
        float *y = output + row * N;
        // 按线程坐标去索引每一行求最大值
        float max = -INFINITY;
        for (int k = 0; k < N; k++) {
            max = fmax(max, x[k]);
        }
        //指数求和
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += expf(x[k] - max);
        }
        //计算softmax值
        float inv_sum = 1.0f / sum;
        for (int k = 0; k < N; k++) {
            y[k] = expf(x[k] - max) * inv_sum;
        }

    }
