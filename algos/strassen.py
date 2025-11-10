# Strassen's Matrix Multiplication Algorithm Implementation in Python
import time

# This implements Strassen's algorithm for matrix multiplication without using external libraries like NumPy.

def add_matrix(A,B):
    l = len(A)
    return [[A[i][j] + B[i][j] for j in range(l) ] for i in range(l)]

def sub_matrix(A,B):
    l = len(A)
    return [[A[i][j] - B[i][j] for j in range(l) ] for i in range(l)]

def split(A,k):
    # a = [[A[i][j] for j in range(k)] for i in range(k)]
    A11 = [row[:k] for row in A[:k]]
    A12 = [row[k:] for row in A[:k]]
    A21 = [row[:k] for row in A[k:]]
    A22 = [row[k:] for row in A[k:]]

    return A11 , A12, A21 , A22 

def pad_matrix(A):
    n = len(A)
    m = 1
    while m < n:
        m *= 2  # find next power of two
    if m == n:
        return A
    # pad with zeros
    padded = [[0]*m for _ in range(m)]
    for i in range(n):
        for j in range(n):
            padded[i][j] = A[i][j]
    return padded


def strassen_matrix_multiply(A,B):
    A = pad_matrix(A)
    B = pad_matrix(B)
    l = len(A)
    if l==1:
        return [[A[0][0]*B[0][0]]]
    
    # Divede matrix
    k = l//2

    A11 , A12 , A21 , A22 = split(A,k)
    B11 , B12 , B21 , B22 = split(B,k)

    

    # Compute all intermediate matrices

    M1 = strassen_matrix_multiply(add_matrix(A11,A22),add_matrix(B11,B22))
    M2 = strassen_matrix_multiply(add_matrix(A21,A22),B11)
    M3 = strassen_matrix_multiply(A11,sub_matrix(B12,B22))
    M4 = strassen_matrix_multiply(A22,sub_matrix(B21,B11))
    M5 = strassen_matrix_multiply(add_matrix(A11,A12),B22)
    M6 = strassen_matrix_multiply(sub_matrix(A21,A11),add_matrix(B11,B12))
    M7 = strassen_matrix_multiply(sub_matrix(A12,A22),add_matrix(B21,B22))

    # Combine the matrices
    C11 = sub_matrix(add_matrix(add_matrix(M1,M4),M7),M5)
    C12 = add_matrix(M3,M5)
    C21 = add_matrix(M2,M4)
    C22 = sub_matrix(add_matrix(add_matrix(M1,M3),M6),M2)
    
    # return the matrices after combining

    C = [(rowa + rowb) for rowa ,rowb in zip(C11,C12)] + [(rowa + rowb) for rowa ,rowb in zip(C21,C22)]

    return [row[:l] for row in C[:l]]
    
if __name__ == "__main__":   
    A = [[1, 2, 3, 4],
         [5, 6, 7, 8],
         [9, 10, 11, 12],
         [13, 14, 15, 16]]
    
    B = [[16, 15, 14, 13],
         [12, 11, 10, 9],
         [8, 7, 6, 5],
         [4, 3, 2, 1]]
    
    start_time = time.time()
    result = strassen_matrix_multiply(A, B)
    end_time = time.time()
    
    for row in result:
        print(row)
    
    print(f"Strassen's algorithm took {end_time - start_time} seconds")


