import numpy as np
import time

def strassen(A, B):

    A = np.array(A)
    B = np.array(B)
    l = A.shape[0]
    if l == 1:
        return np.array([[A[0][0] * B[0][0]]])
    
    k = l//2
    # Dividing the matrices into quarters
    A11 , A12 , A21 , A22  = A[:k , :k] , A[:k , k:] ,A[k: , :k] ,A[k: , k:]
    B11 , B12 , B21 , B22  = B[:k , :k] , B[:k , k:] ,B[k: , :k] ,B[k: , k:]

    # Intermediate matrices
    M1 = strassen(A11 + A22 , B11 + B22)
    M2 = strassen(A21 + A22 , B11)
    M3 = strassen(A11 , B12 - B22)
    M4 = strassen(A22 , B21 - B11)
    M5 = strassen(A11 + A12 , B22)
    M6 = strassen(A21 - A11 , B11 + B12)
    M7 = strassen(A12 - A22 , B21 + B22)

    # Combined matrices
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6

    R = np.zeros(A.shape)
    # Combining the quarters into a single result matrix
    R[:k , :k] , R[:k , k:] ,R[k: , :k] , R[k: , k:]= C11 , C12 , C21 , C22

    return R

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
    result = strassen(A, B)
    end_time = time.time()
    
    for row in result:
        print(row)
    
    print(f"Strassen's algorithm took {end_time - start_time} seconds")


