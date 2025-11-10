import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def plotter(csv_file = "results/timing.csv" , output_image = "results/perfomance_graph.png"):


    # check whether the file exist if not create
    Path(output_image).parent.mkdir(parents = True , exist_ok=True)
    
    # make a dataframe for the csv file
    data = pd.read_csv(csv_file)

    # Plot the graph according to the methods given in the csv file
    for method in data['Method'].unique():
        method_data = data[data['Method'] == method]
        plt.plot(method_data['Matrix Size'], method_data['Time (s)'], marker='o', label=method)
        

    plt.title("Matrix Multiplication Perfomance Comparson")
    plt.xlabel("Matrix size")
    plt.ylabel("Time taken(sec)")
    plt.grid(True)
    plt.tight_layout
    plt.legend()

    #  save and show plot
    plt.savefig(output_image)
    plt.show()
    