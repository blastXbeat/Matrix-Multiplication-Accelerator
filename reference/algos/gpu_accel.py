import cupy as cp
import time


def warm_up():
    # Warm up the GPU to avoid initial latency affecting timing
    A = cp.ones((10,10))
    B = cp.ones((10,10))
    _ = cp.dot(A,B)

def gpu_mat_mul(A,B):
    warm_up()
    A = cp.array(A)
    B = cp.array(B)
    
    # Ensure all previous GPU operations are complete
    cp.cuda.Stream.null.synchronize()

    result = cp.dot(A,B)

    # convert back to numpy array before returning
    return cp.asnumpy(result)

if __name__ == "__main__":   
    # Example usage
    A = [[1, 2, 3, 4],
         [5, 6, 7, 8],
         [9, 10, 11, 12],
         [13, 14, 15, 16]]
    
    B = [[16, 15, 14, 13],
         [12, 11, 10, 9],
         [8, 7, 6, 5],
         [4, 3, 2, 1]]
    
    start_time = time.perf_counter()
    result = gpu_mat_mul(A, B)
    end_time = time.perf_counter()
    
    for row in result:
        print(row)
    
    print(f"GPU mat mul took {end_time - start_time} seconds")

