#include <cuda_runtime.h>
#include <time.h>
#include <cmath>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

struct Matrix {
    int rows;
    int cols; 
    float* data;
};

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

void matrix_mul_CPU(const Matrix M, const Matrix N, Matrix out)
{
    assert(M.cols == N.rows && "Matrix M columns must match Matrix N rows");
    assert(out.rows == M.rows && "Matrix out rows must match Matrix M rows");
    assert(out.cols == N.cols && "Matrix out columns must match Matrix N columns");
    for (int row = 0; row < M.rows; row++)
    {
        for (int col = 0; col < N.cols; col++)
        {

            float out_value = 0;
            for (int k = 0; k < M.cols; ++k) 
            {
                out_value += M.data[row*M.cols + k]*N.data[k*N.cols+col];
            }
    
            out.data[row*N.cols + col] = out_value;
        }
    }
}

__global__ void matrix_mul_GPU(const Matrix M, const Matrix N, Matrix out) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M.rows && col < N.cols) {
        float out_value = 0;
        for (int k = 0; k < M.cols; ++k) {
            out_value += M.data[row * M.cols + k] * N.data[k * N.cols + col];
        }
        out.data[row * N.cols + col] = out_value;
    }
}