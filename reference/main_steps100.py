import reference.algos as algos
import reference.utils as utils
import time

def main():
    # Main function to run benchmarking and plotting for selected matrix multiplication algorithms
    while True:

        # Dictionary mapping method names to their corresponding functions
        Methods = {
            "Naive method": algos.normal_method,
            "Strassen method": algos.strassen_method,
            "Block method": algos.block_method,
            "Numpy method": algos.numpy_method,
            "Parallel CPU method": algos.parallel_method,
            "GPU CuPy method": algos.gpu_cupy_method
        }

        # Selecting the algos to plot benchmark
        for num , method in enumerate(Methods.keys()):
            print(f"{num}: {method}")

        

        method = input("Select method (Give the corresponding numbers): ").split()
        # Validation for method selection
        if (
            len(method) == 0
            or not all(m.isdigit() for m in method)
            or not all(0 <= int(m) < len(Methods) for m in method)
        ):
            print("-->!!! Invalid selection. Please enter valid method numbers.")
            continue

        N = 1000  # Maximum matrix size

        
        otuput_name ='_'.join([list(Methods.keys())[int(m)].replace(" " , "_") for m in method]) 

        start = time.perf_counter()
        # Benchmarking loop
        for n in range(1,N+1,100):
            A = utils.generate_mat(n , low = 0 , high = 100 , seed = 42 , dtype = float)
            B = utils.generate_mat(n , low = 0 , high = 100 , seed = 24 , dtype = float)
            
            # Iterating through selected methods
            for m in method:   

                method_name = list(Methods.keys())[int(m)]
                func = Methods[method_name]

                result , elapsed_time = utils.benchmark(func , A , B)

                utils.log_result(method_name , n , elapsed_time , output_dir = f"results/data/{otuput_name}" + f"_steps100_{N}.csv")
            
                

        # Plotting the results
        utils.plotter( csv_file= f"results/data/{otuput_name}" + f"_steps100_{N}.csv" , output_image = f"results/plots/{otuput_name }" + f"_steps100_{N}.png")
        finish = time.perf_counter()
        print(f"Total benchmarking time: {finish - start} seconds") 
        print("Benchmarking completed and results plotted. Timing log and performance graph saved.")

        # Prompt to continue or exit
        cont = input("Do you want to continue? (y/n): ")
        if cont.lower() != 'y':
            break   

        

if __name__ == "__main__":
    # Entry point of the program .
    # During multiprocessing the spawned child processes will not execute main() .
    main()

