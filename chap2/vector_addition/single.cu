#include <stdio.h>
#include <cuda_runtime.h>
#include <time.h>

#define X 1000000000


__global__
void vecAddKernel(float* A, float* B, float* C, int n)
{
    int i = threadIdx.x + blockDim.x*blockIdx.x;
    if (i < n) 
    {
        C[i] = A[i] + B[i];
    }
}

void vecAdd_CPU(float* A_h,float* B_h,float* C_h, int n)
{
    for (int i = 0; i < n; ++i)
    {
        C_h[i] = A_h[i] + B_h[i];
    }
}

void vecFill(float* V, int n)
{
    for (int i = 0; i < n; i++) {
        V[i] = i*0.1f;  
    }
}

float vecSum(float *V, int n)
{
    float sum = 0;
    for (int i = 0; i < n; i++)
    {
        sum += V[i];
    }
    return sum;
}

void vecAdd(float* A_h,float* B_h,float* C_h, int n)
{
    float *A_d, *B_d, *C_d;
    int size = n*sizeof(float);
    cudaMalloc((void **) &A_d, size);
    cudaMalloc((void **) &B_d, size);
    cudaMalloc((void **) &C_d, size);

    cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice);

    vecAddKernel<<<ceil(n/256.0), 256>>>(A_d, B_d, C_d, n);
    cudaDeviceSynchronize();    // Im adding this but I belive the dependencies 
                                // will already lock the main thread
    cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost);


    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
    
}

int main() 
{
    float *A_h, *B_h, *C_h;

    A_h = (float*)malloc(X * sizeof(float));
    B_h = (float*)malloc(X * sizeof(float));
    C_h = (float*)malloc(X * sizeof(float));

    if (A_h == NULL || B_h == NULL || C_h == NULL) {
        printf("Failed to allocate host memory!\n");
        return 1;
    }

    vecFill(A_h,X);
    vecFill(B_h,X);

    clock_t start = clock();
    vecAdd_CPU(A_h, B_h, C_h, X);
    clock_t end = clock();
    float time = (float)(end - start) / CLOCKS_PER_SEC * 1000;

    float sum = vecSum(C_h,X);
    printf("C sum is: %f\n", sum);
    fflush(NULL);
    printf("CPU time: %.2f ms\n", time);


    start = clock();
    vecAdd(A_h,B_h,C_h,X);
    end = clock();
    time = (float)(end - start) / CLOCKS_PER_SEC * 1000;
    
    sum = vecSum(C_h,X);
    printf("C sum gpu is: %f\n", sum);
    fflush(NULL);
    printf("GPU time: %.2f ms\n", time);


    free(A_h);
    free(B_h);
    free(C_h);
    return 0;
}