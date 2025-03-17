#include <cuda_runtime.h>
#include <time.h>

#include <stdio.h>
#include <stdlib.h>
#include "commons/profiler.cpp"
#define TILE_WIDTH 32

void read_matrix(const char *filename, int rows, int cols, float *matrix) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        perror("Error opening file");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < rows * cols; i++) {
        if (fscanf(file, "%f", &matrix[i]) != 1) {
            perror("Error reading file");
            fclose(file);
            exit(EXIT_FAILURE);
        }
    }

    fclose(file);
}
__global__ void matrixMulKernel(float* M, float* N, float* P, int Width)
{

    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;

    float Pvalue = 0;
    for (int ph = 0; ph < (Width + TILE_WIDTH - 1)/TILE_WIDTH; ++ph) 
    {
        if ((Row < Width) && (ph*TILE_WIDTH+tx) < Width )
        {
            Mds[ty][tx] = M[Row*Width + ph*TILE_WIDTH + tx]; 
        }
        else {Mds[ty][tx] = 0.0f;}
        if ((Col < Width) && (ph*TILE_WIDTH+ty) < Width )
        {
            Nds[ty][tx] = N[(ph*TILE_WIDTH + ty)*Width + Col];
        }
        else {Nds[ty][tx] = 0.0f;}
        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k)
        {
            Pvalue += Mds[ty][k] * Nds[k][tx];
        }
        __syncthreads();
    }
    
    if ((Row < Width) && (Col < Width))
    {
        P[Row*Width + Col] = Pvalue;
    }

}

void run_kernel_tiled(float* A, float* B, float* C, int width)
{
    float *A_d, *B_d, *C_d;
    int size_bytes = width*width*sizeof(float);
    cudaMalloc((void **) &A_d, size_bytes);
    cudaMalloc((void **) &B_d, size_bytes);
    cudaMalloc((void **) &C_d, size_bytes);

    cudaMemcpy(A_d, A, size_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, size_bytes, cudaMemcpyHostToDevice);


    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH,1);
    dim3 dimGrid((width + TILE_WIDTH - 1) / TILE_WIDTH, (width + TILE_WIDTH - 1) / TILE_WIDTH, 1);


    matrixMulKernel<<<dimGrid, dimBlock>>>(A_d, B_d, C_d, width);
    cudaDeviceSynchronize();    // Im adding this but I belive the dependencies 
                                // will already lock the main thread
    cudaMemcpy(C, C_d, size_bytes, cudaMemcpyDeviceToHost);
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);

}

bool matrices_are_almost_equal(float *A, float*B, int n, int m, double theshold = 1e-6) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            if (fabs(A[i * j + j] - B[i * j + j]) > theshold) {
                printf("Matrices do not match at position (%d, %d) -- %f > %lf\n", i, j, fabs(A[i * j + j] - B[i * j + j]), theshold);
                return false;  
            }
        }
    }
    printf("Matrices are almost equal\n");
    return true;  
}

void save_matrix(const char *filename, float *matrix, int rows, int cols) {
    FILE *file = fopen(filename, "w");
    if (file == NULL) {
        printf("Error opening file: %s\n", filename);
        exit(1);
    }

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            fprintf(file, "%.8f ", matrix[i * cols + j]);
        }
        fprintf(file, "\n");
    }

    fclose(file);
}

__global__
void matrix_mul_GPU(float* M, float* N, float* P, int width)
{
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    if (col < width && row < width)
    {
        float Pvalue = 0;
        for (int k = 0; k < width; ++k) 
        {
            Pvalue += M[row*width + k]*N[k*width+col];
        }
    
        P[row*width + col] = Pvalue;
    }
}

void run_kernel_non_tiled(float* A, float* B, float* C, int width)
{
    float *A_d, *B_d, *C_d;
    int size_bytes = width*width*sizeof(float);
    cudaMalloc((void **) &A_d, size_bytes);
    cudaMalloc((void **) &B_d, size_bytes);
    cudaMalloc((void **) &C_d, size_bytes);

    cudaMemcpy(A_d, A, size_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, size_bytes, cudaMemcpyHostToDevice);


    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH,1);
    dim3 dimGrid((width + TILE_WIDTH - 1) / TILE_WIDTH, (width + TILE_WIDTH - 1) / TILE_WIDTH, 1);


    matrix_mul_GPU<<<dimGrid, dimBlock>>>(A_d, B_d, C_d, width);
    cudaDeviceSynchronize();    // Im adding this but I belive the dependencies 
                                // will already lock the main thread
    cudaMemcpy(C, C_d, size_bytes, cudaMemcpyDeviceToHost);
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);

}



int main()
{   

    int size = 5000;
    int warmup = 10;
    int runs = 10;
    float similarity_treshold = 1;
    int n = size, m = size, i = size, j = size;

    float *A = (float *)malloc(n * m * sizeof(float));
    float *B = (float *)malloc(i * j * sizeof(float));
    float *C = (float *)malloc(n * j * sizeof(float));
    float *C_out = (float *)malloc(n * j * sizeof(float));

    read_matrix("matrix_A.txt", n, m, A);
    read_matrix("matrix_B.txt", i, j, B);
    read_matrix("matrix_C.txt", n, j, C);
    printf("First item C: %.8f\n", C[0]);

    {
        for (int k = 0; k < warmup; k++) {
            run_kernel_tiled(A, B, C_out, n);
        }
        
        u64 start = ReadCPUTimer();
        for (int k = 0; k < runs; k++) {
            run_kernel_tiled(A, B, C_out, n);
        }
        u64 elapsed = ReadCPUTimer() - start;
        f64 timems = 1000.0f * (f64)elapsed / (f64)EstimateCPUTimerFreq();
        printf("Tiled avarage time / cycles: %0.4fms / %llu\n", timems / (f64)runs, elapsed);

        matrices_are_almost_equal(C,C_out,size,size,similarity_treshold);
        
        for (int k = 0; k < warmup; k++) {
            run_kernel_non_tiled(A, B, C_out, n);
        }

        start = ReadCPUTimer();
        for (int k = 0; k < runs; k++) {
            run_kernel_non_tiled(A, B, C_out, n);
        }
        elapsed = ReadCPUTimer() - start;
        timems = 1000.0f * (f64)elapsed / (f64)EstimateCPUTimerFreq();
        printf("Non tiled avarage time / cycles: %0.4fms / %llu\n", timems / (f64)runs, elapsed);
        matrices_are_almost_equal(C,C_out,size,size,similarity_treshold);
    }
    return 0;
}
