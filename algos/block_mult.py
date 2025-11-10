import numpy as np
import time

def padding(A):
    # This is done so that evn rectangular matrix can be handled by block method
    i , j = A.shape[0] , A.shape[1]
    m=1
    while m<max(i,j):
        # Finding the next power of 2
        m *= 2
    
    R = np.zeros((m,m))
    R[:i , :j] = A
    return R , i , j

def block_matrix_mult(A , B , is_padded = False):
    if not is_padded:
        A , ai , aj  = padding(np.array(A))
        B , bi , bj = padding(np.array(B))
    else:
        ai , aj = A.shape
        bi , bj = B.shape
    

    l = A.shape[0]
    if l == 2:
        a = A[0,0]*B[0,0] + A[0,1]*B[1,0]
        b = A[0,0]*B[0,1] + A[0,1]*B[1,1]
        c = A[1,0]*B[0,0] + A[1,1]*B[1,0]
        d = A[1,0]*B[0,1] + A[1,1]*B[1,1]
        return np.array([[a , b] , [c , d]])   
    elif l ==1:
        return np.array([[A[0,0]*B[0,0]]])     
    
    k = l//2

    # Dividing the matrices into quarters

    A11 , A12 , A21 , A22  = A[:k , :k] , A[:k , k:] ,A[k: , :k] ,A[k: , k:]
    B11 , B12 , B21 , B22  = B[:k , :k] , B[:k , k:] ,B[k: , :k] ,B[k: , k:]

    R = np.zeros_like(A)

    # Calculating the quarters of resultant matrix

    R[:k , :k] = block_matrix_mult(A11 , B11 , is_padded= True) + block_matrix_mult(A12 , B21 , is_padded= True)
    R[:k , k:] = block_matrix_mult(A11 , B12 , is_padded= True) + block_matrix_mult(A12 , B22 , is_padded= True)
    R[k: , :k] = block_matrix_mult(A21 , B11 , is_padded= True) + block_matrix_mult(A22 , B21 , is_padded= True)
    R[k: , k:] = block_matrix_mult(A21 , B12 , is_padded= True) + block_matrix_mult(A22 , B22 , is_padded= True)

    return R[:ai , :bj]
    
if __name__ == "__main__":   
    #  Example usage
    A = [[1, 2, 3, 4],
         [5, 6, 7, 8],
         [9, 10, 11, 12],
         [13, 14, 15, 16]]
    
    B = [[16, 15, 14, 13],
         [12, 11, 10, 9],
         [8, 7, 6, 5],
         [4, 3, 2, 1]]
    
    start_time = time.time()
    result = block_matrix_mult(A, B)
    end_time = time.time()
    
    for row in result:
        print(row)
    
    print(f"Block multiplication algorithm took {end_time - start_time} seconds")    