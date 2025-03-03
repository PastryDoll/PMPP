#include "utils.cu"

int main()
{
    int n = 1000, m = 2000, i = 2000, j = 500;
    float *A = (float *)malloc(n * m * sizeof(float));
    float *B = (float *)malloc(i * j * sizeof(float));
    float *C = (float *)malloc(n * j * sizeof(float));
    float *C_out = (float *)malloc(n * j * sizeof(float));

    Matrix m_A = {.rows = n, .cols = m, .data = A};
    Matrix m_B = {.rows = i, .cols = j, .data = B};
    Matrix m_C = {.rows = n, .cols = j, .data = C};
    Matrix m_C_out = {.rows = n, .cols = j, .data = C_out};

    read_matrix("matrix_A.txt", n, m, A);
    read_matrix("matrix_B.txt", i, j, B);
    read_matrix("matrix_C.txt", n, j, C);

    matrix_mul_CPU(m_A,m_B,m_C_out);
    matrices_are_almost_equal(C,C_out,n,j,0.2);
    save_matrix("C_test.txt", C_out, n, j);

    return 0;
}