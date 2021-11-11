#include <math.h>
#include <stdlib.h>
#include <stdio.h>

const double EPSILON = 1.0e-15;
const double a = 1.23;
const double b = 2.34;
const double c = 3.57;
__global__ void add(const double *x, const double *y, double *z, int N);
void check(const double *z, const int N);

int main(void)
{
    const int N = 100000000;
    const int M = sizeof(double) * N;
    double *xGPU, *yGPU, *zGPU;

    int blockDimx = 1024;
    int gridDimx = ceil(1.0 *N / blockDimx);
    // printf("%d", N / blockDimx);

    cudaMalloc((void**)&xGPU, M);
    cudaMalloc((void**)&yGPU, M);
    cudaMalloc((void**)&zGPU, M);
    
    double *x = (double*) malloc(M);
    double *y = (double*) malloc(M);
    double *z = (double*) malloc(M);


    for (int n = 0; n < N; ++n)
    {
        x[n] = a;
        y[n] = b;
    }

    cudaMemcpy(xGPU, x, M, cudaMemcpyHostToDevice);
    cudaMemcpy(yGPU, y, M, cudaMemcpyHostToDevice);
    cudaMemcpy(zGPU, z, M, cudaMemcpyHostToDevice);

    add<<<gridDimx, blockDimx>>>(xGPU, yGPU, zGPU, N);

    // cudaDeviceSynchronize();

    cudaMemcpy(z, zGPU, M, cudaMemcpyDeviceToHost);
    check(z, N);

    free(x);
    free(y);
    free(z);
    return 0;
}

__global__ void add(const double *x, const double *y, double *z, int N)
{
    int n = blockDim.x * blockIdx.x + threadIdx.x;
    // z[i*blockDim.x + j] = x[i*blockDim.x + j] + y[i*blockDim.x + j];
    if(n < N){
        z[n] = x[n] + y[n];
    }
}

void check(const double *z, const int N)
{
    bool has_error = false;
    for (int n = 0; n < N; ++n)
    {
        if (fabs(z[n] - c) > EPSILON)
        {
            printf("%d\n", n);
            has_error = true;
        }
    }
    printf("%s\n", has_error ? "Has errors" : "No errors");
}

