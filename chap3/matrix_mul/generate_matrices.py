import numpy as np

def generate_and_save_matrices(n, m, i, j):
    A = (np.random.rand(n, m) * np.array([10])).astype(np.float32)
    B = (np.random.rand(i, j) * np.array([10])).astype(np.float32)
    C = (np.zeros(shape= (n, j)) * np.array([0])).astype(np.float32)

    if m != i:
        raise ValueError(f"Incompatible dimensions: A is {n}x{m} and B is {i}x{j}")

    np.dot(A, B, out=C)

    np.savetxt('matrix_A.txt', A.astype(np.float32), fmt='%.8f')
    np.savetxt('matrix_B.txt', B.astype(np.float32), fmt='%.8f')
    np.savetxt('matrix_C.txt', C.astype(np.float32), fmt='%.8f')

    print("Matrices saved to files: matrix_A.txt, matrix_B.txt, matrix_C.txt")

if __name__ == "__main__":
    n, m, i, j = 1000,2000,2000,500
    generate_and_save_matrices(n, m, i, j)
