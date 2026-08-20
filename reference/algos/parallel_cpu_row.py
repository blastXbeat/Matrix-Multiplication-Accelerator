import numpy as np
from multiprocessing import Pool
import time



def parallel_matrix_multiply(A,B):
    # Multiplying matrix A and B in parallel by distributing rows of A to different processes
    A=np.array(A)
    B=np.array(B)

    with Pool() as p:
        # Creating a list of tasks where each task is a row of A and the entire matrix B
        l = [(A[i,:],B) for i in range(A.shape[0])]

        result = p.starmap(np.dot , l)
    return np.array(result)
    
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