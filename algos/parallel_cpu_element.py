import numpy as np
from multiprocessing import Pool 
import time

def compute_val(A_row , B_col):
    # Compute single element of the result matrix
    return sum(a*b for a,b in zip(A_row, B_col))

def parallel_matrix_multiply(A, B):
    A  = np.array(A)
    B  = np.array(B)

    if A.shape[1] != B.shape[0]:
        raise ValueError("Incompatible matrix dimensions for multiplication")
    
    # Transpose B to facilitate easy column access
    B_t = B.T

    n_rows , n_cols = A.shape[0] , B.shape[1]


    # create tasks for each element in the result matrix
    tasks = [((A[i,:]) , B_t[j,:]) for i in range(n_rows) for j in range(n_cols)]

    with Pool() as p:
        # Map tasks to worker processes

        result = p.starmap(compute_val , tasks)

    C = np.array(result).reshape(n_rows,n_cols)

    return C    

# Example usage
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
    result = parallel_matrix_multiply(A, B)
    end_time = time.time()
    
    for row in result:
        print(row)
    
    print(f"Multiprocessing algorithm took {end_time - start_time} seconds")


    

    
