import numpy as np
import time

# simple numpy matrix multiplication
def numpy_mat_mult(A , B):
    A = np.array(A)
    B = np.array(B)
    return A @ B

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
    result = numpy_mat_mult(A, B)
    end_time = time.time()
    
    for row in result:
        print(row)
    
    print(f"Numpy's innate algorithm took {end_time - start_time} seconds")