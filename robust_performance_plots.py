import json
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys

if __name__ == "__main__":
    # Read the name of file from the command line arguments
    if len(sys.argv) < 2:
        print("Usage: python robust_performance_plots.py <file_name>")
        sys.exit(1) 
    file_name = sys.argv[1]
    model_name = file_name.split("/")[-2].split(".")[0]
    # Convert json data to lists of floats
    def convert_to_floats(data):
        if isinstance(data, list):
            return [convert_to_floats(item) for item in data]
        elif isinstance(data, dict):
            return {key: convert_to_floats(value) for key, value in data.items()}
        elif isinstance(data, str) and data.replace('.', '', 1).isdigit():
            return float(data)
        else:
            return data
        

    # Load the data from the JSON file
    with open(file_name, 'r') as file:
        data = json.load(file)


    # Convert the data to a DataFrame
    for key in data:
        data[key] = eval(data[key])  # Convert string representations of lists to actual lists
        data[key] = np.abs(data[key])
    df = pd.DataFrame(data)
    # These are the columns that we want to plot
    # rl_performance_single_pomdp
    # extracted_fsc_performance
    # family_performance

    columns_to_plot = [
        "rl_performance_single_pomdp",
        "extracted_fsc_performance",
        "family_performance"]
    
    # Create a new DataFrame with the columns to plot
    df_to_plot = df[columns_to_plot].copy()
    # Rename the columns for better readability
    df_to_plot.columns = [
        "RL Performance i+1 POMDPs",
        "Extracted FSC Performance on i+1 POMDPs",
        "Family Performance (Worst Case)"
    ]

    
    plt.figure(figsize=(12, 6))
    # Melt the DataFrame to have a long format for seaborn
    df_to_plot.plot()
    goal = "(Minimization)" if "obstacle" in model_name else "(Maximization)"
    plt.xlabel('i-th POMDP')
    plt.ylabel('Value')
    plt.grid(True)
    plt.legend(title='Type')
    plt.tight_layout()
    save_path = file_name.replace(".json", "_performance.png")
    plt.savefig(save_path)

