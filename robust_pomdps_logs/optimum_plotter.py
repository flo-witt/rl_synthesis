import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def parse_log_file(file_path):
    """
    Parses the log file and returns a DataFrame with the relevant data.
    """
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('optimum'):
                parts = line.split()
                optimum = float(parts[1])
                data.append(optimum)
    return pd.DataFrame(data, columns=['optimum'])

def plot_optimums(log_file_path):
    """
    Plots the optimum values from the log file.
    """
    df = parse_log_file(log_file_path)
    model_name = log_file_path.replace('.log', '')
    
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x=df.index, y='optimum', marker='o')
    
    plt.title(f'The Worst Optimum POMDP Value Over Iterations for {model_name}')

    if df['optimum'].shape[0] > 0: # Plot vertical line to indicate the last optimum from first phase
        plt.axvline(0.5, color='red', linestyle='--', label='First Phase End')
    if df['optimum'].shape[0] > 16: # Plot vertical line to indicate the last optimum from the second phase
        plt.axvline(16.5, color='blue', linestyle='--', label='Second Phase End')

        

    plt.xlabel('Iteration')
    plt.ylabel('Optimum Value')
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    
    save_path = log_file_path.replace('.log', '_optimum_plot.png')
    plt.savefig(save_path)

if __name__ == "__main__":
    # Use the first argument as the log file path
    import sys
    if len(sys.argv) != 2:
        print("Usage: python optimum_plotter.py <log_file_path>")
    else:
        log_file_path = sys.argv[1]
        plot_optimums(log_file_path)