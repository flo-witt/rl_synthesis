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


def plot_returns(returns, title="Returns Avoid", constant_return=None):
    """Plot returns and reach probabilities."""
    df = pd.DataFrame({
        'Returns': returns,
    })
    plt.figure(figsize=(6, 6))
    sns.lineplot(data=df['Returns'], label='Returns (minimization)')
    # Add constant line for returns
    if constant_return is not None:
        plt.axhline(y=constant_return, color='r',
                    linestyle='--', label='Worst-case from rfPG')
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
        sns.lineplot(data=df['Returns'],
                     label=f'Robust Performance {names[i]}')

    plt.title(title)
    plt.xlabel('i-th POMDP added to the training')
    plt.ylim(bottom=0, top=800)
    plt.ylabel('Values')
    if constant_return is not None:
        plt.axhline(y=constant_return, color='r',
                    linestyle='--', label='Worst-case from rfPG')
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


def plot_family_performance():
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


def plot_some_metrics(metric_dict, metric_name=None, constant_dict=None, data_names=None):
    """
    Plot all metrics, with optional twin y-axis for specified metrics.

    Parameters:
    - twin_y_metrics: List of metric names to plot on the twin y-axis.
    """
    fig, ax1 = plt.subplots(figsize=(12, 10))
    colors = plt.cm.tab10.colors
    line_styles = ['-', '--', ':', '-.']
    markers = ['x', 'o', 's', '^', 'D']

    # Hlavní osa (ax1)
    for i, (m, metric_data_list) in enumerate(metric_dict.items()):
        if m == "DBSCAN Clusters":
            continue  # Tyto metriky vykreslíme na druhou osu
        period = metric_name[m] if isinstance(metric_name, dict) else 1
        for j, metric_data in enumerate(metric_data_list):
            df = pd.DataFrame({m: metric_data})
            color = colors[j % len(colors)]
            label = f"{data_names[j]} ({m})" if data_names is not None else f"{m}_{j}"
            if period > 1:
                indices = np.arange(0, len(df[m]) * period, period)
                ax1.plot(
                    indices,
                    df[m],
                    label=label,
                    color=color,
                    linestyle='None',
                    marker=markers[i % len(markers)],
                    markersize=8,
                    mec=color,
                    mew=2,
                )
            else:
                sns.lineplot(
                    data=df,
                    x=df.index,
                    y=m,
                    ax=ax1,
                    label=label,
                    color=color,
                    linestyle=line_styles[i % len(line_styles)],
                    linewidth=2
                )

    # Druhá osa (ax2)
    ax2 = ax1.twinx()
    for i, (m, metric_data_list) in enumerate(metric_dict.items()):
        print(f"Processing metric: {m}")
        if not m == "DBSCAN Clusters":
            continue
        period = metric_name[m] if isinstance(metric_name, dict) else 1
        for j, metric_data in enumerate(metric_data_list):
            df = pd.DataFrame({m: metric_data})
            color = colors[(j + 3) % len(colors)]  # Jiná barva pro druhou osu
            label = f"{data_names[j]} ({m})" if data_names is not None else f"{m}_{j}"
            if period > 1:
                indices = np.arange(0, len(df[m]) * period, period)
                ax2.plot(
                    indices,
                    df[m],
                    label=label,
                    color=color,
                    linestyle='None',
                    marker=markers[i % len(markers)],
                    markersize=8,
                    mec=color,
                    mew=2,
                )
            else:
                sns.lineplot(
                    data=df,
                    x=df.index,
                    y=m,
                    ax=ax2,
                    label=label,
                    color=color,
                    linestyle=line_styles[i % len(line_styles)],
                    linewidth=2
                )

    # Legenda a popisky
    handles1, labels1 = ax1.get_legend_handles_labels()
    if constant_dict is not None:
        for key, value in constant_dict.items():
            print(key)
            line = ax1.axhline(y=value, color='gray', linestyle='--', label=f'Constant {key}')
            handles1.append(line)
            labels1.append(f'Constant {key}')
    handles2, labels2 = ax2.get_legend_handles_labels() if "DBSCAN Clusters" in metric_name else ([], [])
    ax1.legend(handles=handles1 + handles2, labels=labels1 + labels2, loc='upper left')
    ax1.set_title("Metrics Plot")
    ax1.set_xlabel('i-th POMDPs added to the training')
    ax1.set_ylabel('Values (main axis)')
    if "DBSCAN Clusters" in metric_name:
        ax2.set_ylabel('Values (twin axis)')
    ax1.grid(True)

    plt.savefig('metrics_plot.png')
    plt.close()


def get_renamer():
    metrics_renamer = {
        "family_performance": "Verified Worst-Case Performance",
        "average_rl_return_subset_simulated": "Empirical RL Subset Simulated",
        "average_extracted_fsc_return_subset_simulated": "Empirical FSC Subset Simulated",
        "nr_of_clusters": "DBSCAN Clusters"
    }
    return metrics_renamer

def plot_single_file(file_path, metric, constant_dict=None, data_names=None):
    """Load a single file and plot the specified metric."""
    result = load_result_from_json(file_path)
    if isinstance(metric, list):
        metrics_dict = {m: ast.literal_eval(result[m]) for m in metric}
    else:
        metrics_dict = {metric: ast.literal_eval(result[metric])}

    # Convert to numpy arrays
    metrics_numpy = {m: np.abs(np.array(numbers, dtype=np.float32)) for m, numbers in metrics_dict.items()}

    # Plot the metrics
    plot_some_metrics(metrics_numpy, metric_name=metric, constant_dict=constant_dict, data_names=data_names)

def get_and_plot_some_metric(file_path, metric, constant_dict=None, data_names=None):
    """Load specific metrics from JSON files and return a dictionary of metrics and their data."""
    if isinstance(file_path, list):
        results = [load_result_from_json(fp) for fp in file_path]
        metrics_dict = {}

        # Pokud je 'metric' slovník {metric_name: period}
        if isinstance(metric, dict):
            for m, period in metric.items():
                metric_data = [ast.literal_eval(result[m]) for result in results]
                metric_numpy = [np.abs(np.array(numbers, dtype=np.float32)) for numbers in metric_data]
                metrics_dict[m] = metric_numpy

            # Přejmenování metrik, pokud je potřeba
            metrics_renamer = get_renamer()
            metrics_dict = {metrics_renamer.get(m, m): v for m, v in metrics_dict.items()}
            metric = {metrics_renamer.get(m, m): period for m, period in metric.items()}

            # Předání slovníku s periodami jako 'metric_name'
            plot_some_metrics(metrics_dict, metric_name=metric, constant_dict=constant_dict, data_names=data_names)
            return metrics_dict
        # Pokud je 'metric' seznam nebo řetězec (původní chování)
        else:
            if isinstance(metric, list):
                for m in metric:
                    metric_data = [ast.literal_eval(result[m]) for result in results]
                    metric_numpy = [np.abs(np.array(numbers, dtype=np.float32)) for numbers in metric_data]
                    metrics_dict[m] = metric_numpy
            else:
                metric_data = [ast.literal_eval(result[metric]) for result in results]
                metric_numpy = [np.abs(np.array(numbers, dtype=np.float32)) for numbers in metric_data]
                metrics_dict[metric] = metric_numpy

            # Přejmenování metrik, pokud je potřeba
            metrics_renamer = get_renamer()
            metrics_dict = {metrics_renamer.get(m, m): v for m, v in metrics_dict.items()}

            # Předání původního 'metric' jako 'metric_name'
            plot_some_metrics(metrics_dict, metric_name=metric, constant_dict=constant_dict, data_names=data_names)
            return metrics_dict
    else:
        # Pokud je 'file_path' jediný soubor, použijeme původní chování
        plot_single_file(file_path, metric, constant_dict=constant_dict, data_names=data_names)
        return None


def main():
    metrics = ["family_performance", "average_rl_return_subset_simulated", "average_extracted_fsc_return_subset_simulated"]
    metrics_with_periods = {
        "family_performance": 1,
        "average_rl_return_subset_simulated": 1,
        "average_extracted_fsc_return_subset_simulated": 1,
        "nr_of_clusters": 1,
        "worst_case_on_subset_rl" : 5,
        "worst_case_on_subset_fsc": 5,    
    }
    get_and_plot_some_metric(["models_robust_subset/avoid/benchmark_stats_43.json",
                              ],
                             metric=metrics_with_periods,
                             constant_dict={"Worst-case from rfPG": 161.0},
                             data_names=["Geometric Without Restart, short trainings",
                                         "Geometric Without Restart, short trainings, shrink and perturb"])
    # reach_probs_numbers = ast.literal_eval(result["reach_probs"])
    # reach_probs_numpy = np.array(reach_probs_numbers, dtype=np.float32)
    # plot_reach_probabilities(reach_probs_numpy)


if __name__ == "__main__":
    main()
