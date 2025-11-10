import time
import csv
from pathlib import Path

# Utility functions for benchmarking matrix multiplication algorithms
def benchmark(func , *args , **kwargs):
    
    start = time.perf_counter()
    result = func(*args , **kwargs)
    end = time.perf_counter()
    elaps = end - start 
    # Returns the result of the function and the elapsed time
    return result , elaps


# Logs the benchmarking result to a CSV file
def log_result(method_nme , matrix_size , elapsed_time , output_dir = "results/timing.csv"):

    # Ensure the output directory exists if not, create it . Inside mkdir use parents=True
    # to create any necessary parent directories as well as exist_ok=True to avoid raising
    # an error if the directory already exists.
    Path(output_dir).parent.mkdir(parents=True , exist_ok=True)

    # Creates a CSV file and writes the header if it doesn't exist
    header =  ["Method", "Matrix Size", "Time (s)"]

    write_header = (not Path(output_dir).exists()) or (Path(output_dir).stat().st_size == 0)
    # not only we must check if file exists but also wether it is empty or not

    with open(output_dir , 'a' , newline="") as csvfile:
        # Writing to csv file
        writer = csv.writer(csvfile)
        if write_header :
            writer.writerow(header)
        
        writer.writerow([method_nme,matrix_size,elapsed_time])



