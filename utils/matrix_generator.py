import numpy as np

def generate_mat(size  , low = 0 , high = 100 , seed = None , dtype = float):
    # to generate a random sqr matrix of specific size and range

    # seeding for generating repeated sequence
    if seed != None :
        np.random.seed(seed)

    return np.random.uniform(low , high , (size,size)).astype(dtype)

if __name__ == "__main__":
    mat = generate_mat(4 , 0 , 10 , seed = 42 , dtype = int)
    print(mat)