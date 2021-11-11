#include<stdio.h>

__global__ void helloFromGpu()
{
    int i = threadIdx.x;
    int k = threadIdx.y;
    int j = blockIdx.x;
    printf("hello world!%d, %d, %d\n", j, i, k);
}

int main()
{
    const dim3 blockSize(2, 4);
    helloFromGpu<<<2, blockSize>>>();
    cudaDeviceSynchronize();
    return 0;
}