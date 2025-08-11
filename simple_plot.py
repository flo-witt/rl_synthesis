import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import json

import ast
import numpy as np

def load_result_from_json(file_path):
    """Load the result from a JSON file."""
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def plot_returns(returns, title="Returns Avoid", constant_return = None):
    """Plot returns and reach probabilities."""
    df = pd.DataFrame({
        'Returns': returns,
    })
    plt.figure(figsize=(6, 6))
    sns.lineplot(data=df['Returns'], label='Returns (minimization)')
    # Add constant line for returns
    if constant_return is not None:
        plt.axhline(y=constant_return, color='r', linestyle='--', label='Worst-case from rfPG')
    plt.title(title)
    plt.xlabel('i-th 100 training steps')
    plt.ylim(bottom=0, top=500) 
    plt.ylabel('Values')
    if constant_return is not None:
        plt.legend(['Returns', 'Worst-case from rfPG'])
    else:
        plt.legend(['Returns'])
    plt.grid(True)
    plt.savefig('returns.png')

def plot_multiple_returns_in_single_figure(returns_list, names, title="Returns Avoid", constant_return=None):
    """Plot multiple returns in a single figure."""
    plt.figure(figsize=(6, 6))
    for i, returns in enumerate(returns_list):
        df = pd.DataFrame({
            'Returns': returns,
        })
        sns.lineplot(data=df['Returns'], label=f'Robust Performance {names[i]}')
    
    plt.title(title)
    plt.xlabel('i-th POMDP added to the training')
    plt.ylim(bottom=0, top=800) 
    plt.ylabel('Values')
    if constant_return is not None:
        plt.axhline(y=constant_return, color='r', linestyle='--', label='Worst-case from rfPG')
    plt.legend()
    plt.grid(True)
    plt.savefig('multiple_returns.png')

def plot_reach_probabilities(reach_probs, title="Reach Probabilities Avoid"):
    """Plot reach probabilities."""
    df = pd.DataFrame({
        'Reach Probabilities': reach_probs,
    })

    plt.figure(figsize=(6, 6))
    sns.lineplot(data=df['Reach Probabilities'], label='Reach Probabilities')
    plt.title(title)
    plt.xlabel('i-th 100 training steps')
    plt.ylabel('Values')
    plt.legend(['Reach Probabilities'])
    plt.grid(True)
    plt.savefig('reach_probabilities.png')

def load_single_family_performance(file_path):
    result = load_result_from_json(file_path)

    # Convert the result strings to numpy arrays
    returns_numbers = ast.literal_eval(result["family_performance"])
    returns_numpy = np.array(returns_numbers, dtype=np.float32)
    returns_numpy = np.abs(returns_numpy)  # Ensure all returns are positive
    return returns_numpy

def main():
    file_path = "models_robust_subset/avoid/benchmark_stats_6.json"
    returns_numpy = load_single_family_performance(file_path)
    file_path = "models_robust_subset/avoid/benchmark_stats_7.json"
    returns_numpy_2 = load_single_family_performance(file_path)
    # Plot returns
    plot_multiple_returns_in_single_figure(
        [returns_numpy, returns_numpy_2],
        names=["Without Restart", "Each Iteration Restart"], title="Verified Worst-Case Avoid (Minimization)",
        constant_return=161.0
    )

    # reach_probs_numbers = ast.literal_eval(result["reach_probs"])
    # reach_probs_numpy = np.array(reach_probs_numbers, dtype=np.float32)
    # plot_reach_probabilities(reach_probs_numpy)

if __name__ == "__main__":
    main()