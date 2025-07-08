import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Data has shape {keys: [value 1, value 2, ...]}
def plot_robustification(data, title="Robustification Plot", xlabel="Algorithm iteration", ylabel="Return", save_path=None):
    """
    Plots the robustification data as a bar plot with error bars.
    
    Parameters:
    - data: Dictionary where keys are hole assignments and values are lists of returns.
    - title: Title of the plot.
    - xlabel: Label for the x-axis.
    - ylabel: Label for the y-axis.
    - save_path: If provided, saves the plot to this path.
    """
    df = pd.DataFrame(data)
    # Are the values in the data minus, or positive?
    is_positive = all(all(value >= 0 for value in values) for values in data.values())


    # Use absolute values for the values in the values

    if not is_positive:

        df = df.applymap(lambda x: np.abs(x) if isinstance(x, (int, float)) else x)
    # Calculate means and standard deviations
    means = df.mean(axis=1)
    stds = df.std(axis=1)
    print("Means:\n", means)
    print("Standard Deviations:\n", stds)
    # Compute worst case performance for each iteration
    print(df)
    if not is_positive:
        worst_case_performance = df.max(axis=1)
    else:
        worst_case_performance = df.min(axis=1)

    # Create lineplot for worst case performance and line with means with standard deviations
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=worst_case_performance, marker='o', label='Worst Case Performance')
    sns.lineplot(data=means, marker='o', label='Mean Performance')
    plt.fill_between(means.index, means - stds, means + stds, color='lightgray', alpha=0.5, label='Standard Deviation')

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    plt.show()

if __name__ == "__main__":
    # Load data from a file from command line arguments
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python robustification_plots.py <data_file>")
        sys.exit(1)
    
    data_file = sys.argv[1]
    with open(data_file, 'r') as f:
        # remove the first line and eval the rest. The format is (x, y, z, w) : [value1, value2, ...]
        data = f.read().splitlines()[1:]
        data = {line.split(':')[0].strip(): eval(line.split(':')[1].strip()) for line in data if ':' in line}
    file_name = data_file.split('/')[-1].split('.')[0]
    model_name = data_file.split('/')[-2]

    plot_robustification(data, title="Robustification Plot", xlabel="Algorithm Iteration", ylabel="Return", save_path=f"robustification_plot_{model_name}_{file_name}.png")

    