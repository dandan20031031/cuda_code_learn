#include <stdio.h>
#include <cuda_runtime.h>

__global__ void kernel()
{
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    printf("Block[%d] Thread[%d] Global[%d]: hello world\n", blockIdx.x, threadIdx.x, global_id);
}

int main()
{
    kernel<<<2,4>>>();
    printf("hello GPU\n");
    cudaDeviceSynchronize();
    return 0;
}