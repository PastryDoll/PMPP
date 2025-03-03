#include <cuda_runtime.h>
#include <time.h>

#include <stdio.h>
#include <stdlib.h>
#include "commons/profiler.cpp"
#define BLUR_KERNEL_SIZE 7 

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

void matrix_mul_CPU(float* M, float* N, float* P, int width)
{
    for (int row = 0; row < width; row++)
    {
        for (int col = 0; col < width; col++)
        {

            float Pvalue = 0;
            for (int k = 0; k < width; ++k) 
            {
                Pvalue += M[row*width + k]*N[k*width+col];
            }
    
            P[row*width + col] = Pvalue;
        }
    }
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
__global__
void matrix_mul_GPU_1D(float* M, float* N, float* P, int width)
{
    int row = blockIdx.x*blockDim.x + threadIdx.x;
    if (row < width)
    {
        for (int c = 0; c < width; ++c)
        {
            float Pvalue = 0;

            for (int r = 0; r < width; ++r) 
            {
                Pvalue += M[row*width + r]*N[r*width + c];
            }

            P[row*width + c] = Pvalue;
        }
    
    }
}
void run_kernel(float* A, float* B, float* C, int width)
{
    float *A_d, *B_d, *C_d;
    int size_bytes = width*width*sizeof(float);
    cudaMalloc((void **) &A_d, size_bytes);
    cudaMalloc((void **) &B_d, size_bytes);
    cudaMalloc((void **) &C_d, size_bytes);

    cudaMemcpy(A_d, A, size_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, size_bytes, cudaMemcpyHostToDevice);


    dim3 dimGrid(ceil(width/32.0),ceil(width/32.0),1);
    dim3 dimBlock(32,32,1);

    matrix_mul_GPU<<<dimGrid, dimBlock>>>(A_d, B_d, C_d, width);
    cudaDeviceSynchronize();    // Im adding this but I belive the dependencies 
                                // will already lock the main thread
    cudaMemcpy(C, C_d, size_bytes, cudaMemcpyDeviceToHost);
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);

}

void run_kernel_1D(float* A, float* B, float* C, int width)
{
    float *A_d, *B_d, *C_d;
    int size_bytes = width*width*sizeof(float);
    cudaMalloc((void **) &A_d, size_bytes);
    cudaMalloc((void **) &B_d, size_bytes);
    cudaMalloc((void **) &C_d, size_bytes);

    cudaMemcpy(A_d, A, size_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, size_bytes, cudaMemcpyHostToDevice);


    dim3 dimGrid(ceil(width/1024.0),1,1);
    dim3 dimBlock(1024,1,1);

    matrix_mul_GPU_1D<<<dimGrid, dimBlock>>>(A_d, B_d, C_d, width);
    cudaDeviceSynchronize();    // Im adding this but I belive the dependencies 
                                // will already lock the main thread
    cudaMemcpy(C, C_d, size_bytes, cudaMemcpyDeviceToHost);
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);

}


int main() 
{
    BeginProfile();

    int size = 1000;
    int warmup = 2;
    int runs = 3;
    int n = size, m = size, i = size, j = size;

    float *A = (float *)malloc(n * m * sizeof(float));
    float *B = (float *)malloc(i * j * sizeof(float));
    float *C = (float *)malloc(n * j * sizeof(float));
    float *C_out = (float *)malloc(n * j * sizeof(float));
    float *C_out_cuda = (float *)malloc(n * j * sizeof(float));
    float *C_out_1dcuda = (float *)malloc(n * j * sizeof(float));

    read_matrix("matrix_A.txt", n, m, A);
    read_matrix("matrix_B.txt", i, j, B);
    read_matrix("matrix_C.txt", n, j, C);
    printf("First item A: %.8f\n", A[0]);
    printf("First item B: %.8f\n", B[0]);
    printf("First item C: %.8f\n", C[0]);

    {
        // Silly and horrible way to handle warmup and profilling.. will change this on next chapter
        BeginProfile();

        for (int k = 0; k < warmup; k++) {
            matrix_mul_CPU(A, B, C_out, size);
        }
    
        for (int k = 0; k < runs; k++) {
            TimeBlock("CPU")
            matrix_mul_CPU(A, B, C_out, size);
        }
    
        for (int k = 0; k < warmup; k++) {
            run_kernel(A, B, C_out_cuda, n);
        }
    
        for (int k = 0; k < runs; k++) {
            TimeBlock("GPU")
            run_kernel(A, B, C_out_cuda, n);
        }
    
        for (int k = 0; k < warmup; k++) {
            run_kernel_1D(A, B, C_out_1dcuda, n);
        }
    
        for (int k = 0; k < runs; k++) {
            TimeBlock("GPU_1D")
            run_kernel_1D(A, B, C_out_1dcuda, n);
        }

        EndAndPrintProfile();

    }


    printf("First item C_out: %.8f\n", C_out[0]);
    printf("Difference between first items: %.8f\n", fabs(C[0] - C_out[0]));
    matrices_are_almost_equal(C,C_out,size,size,0.01);
    matrices_are_almost_equal(C,C_out_cuda,size,size,0.01);
    matrices_are_almost_equal(C_out,C_out_cuda,size,size,0.01);
    save_matrix("C_CPU.txt", C_out, size, size);
    save_matrix("C_GPU.txt", C_out_cuda, size, size);
    save_matrix("C_GPU_1D.txt", C_out_1dcuda, size, size);



    free(A);
    free(B);
    free(C);
    free(C_out);
    return 0;
}
