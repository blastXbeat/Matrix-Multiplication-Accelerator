# Naive Method of Matrix Multiplication
import time

def naive_matrix_multiplication(A, B):
    l=len(A)
    r=[[0]*l for _ in range(l)]
    # loop to iterate through rows of A
    for i in range(l):
        # loop to iterate through columns of B
        for j in range(l):
            for k in range(l):
                r[i][j] += A[i][k] * B[k][j]
    return r

# Example usage:
if __name__ == "__main__":
    A = [[1, 2, 3],
         [4, 5, 6],
         [7, 8, 9]]

    B = [[9, 8, 7],
         [6, 5, 4],
         [3, 2, 1]]

    start_time = time.time()
    result = naive_matrix_multiplication(A, B)
    end_time = time.time()

    print("Resultant Matrix:")
    for row in result:
        print(row)
    print(f"Time taken: {end_time - start_time} seconds")