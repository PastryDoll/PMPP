#include <cuda_runtime.h>
#include <time.h>

#include <stdio.h>
#include <stdlib.h>
#define STB_IMAGE_IMPLEMENTATION
#include "commons/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "commons/stb_image_write.h"
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
    printf("Matrices are almost equal");
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
            fprintf(file, "%.2f ", matrix[i * cols + j]);
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



int main() 
{
    BeginProfile();

    int size = 100;
    int n = size, m = size, i = size, j = size;

    float *A = (float *)malloc(n * m * sizeof(float));
    float *B = (float *)malloc(i * j * sizeof(float));
    float *C = (float *)malloc(n * j * sizeof(float));
    float *C_out = (float *)malloc(n * j * sizeof(float));

    read_matrix("matrix_A.txt", n, m, A);
    read_matrix("matrix_B.txt", i, j, B);
    read_matrix("matrix_C.txt", n, j, C);
    printf("First item A: %.8f\n", A[0]);
    printf("First item B: %.8f\n", B[0]);
    printf("First item C: %.8f\n", C[0]);
    {
        TimeBlock("CPU")
        matrix_mul_CPU(A,B,C_out,size);
    }

    printf("First item C_out: %.8f\n", C_out[0]);
    printf("Difference between first items: %.8f\n", fabs(C[0] - C_out[0]));
    matrices_are_almost_equal(C,C_out,size,size,0.001);
    save_matrix("C_CPU.txt", C_out, size, size);


    EndAndPrintProfile();

    free(A);
    free(B);
    free(C);
    free(C_out);
    return 0;
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
