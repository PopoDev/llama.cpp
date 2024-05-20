import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import argparse

def load_results(results_dir='results/ngl', hardware='RTX_2080'):
    data = []
    results_dir += '/' + hardware
    for filename in os.listdir(results_dir):
        if filename.endswith('.json'):
            with open(os.path.join(results_dir, filename), 'r') as f:
                data.append(json.load(f))
    return data

def plot_results(data):
    results_dir = 'results'
    df = pd.DataFrame(data)
    sns.set_theme(style="whitegrid")

    fig, ax1 = plt.subplots(figsize=(12, 6))

    color_gpu_ram = 'tab:green'
    color_cpu_ram = 'tab:red'
    ax1.set_xlabel('Number of GPU Layers')
    ax1.set_ylabel('RAM (MiB)')
    
    sns.lineplot(data=df, x='n_gpu_layers', y='gpu_ram_mib', marker='o', ax=ax1, color=color_gpu_ram, label='GPU RAM (MiB)')
    sns.lineplot(data=df, x='n_gpu_layers', y='cpu_ram_mib', marker='o', ax=ax1, color=color_cpu_ram, label='CPU RAM (MiB)')
    ax1.tick_params(axis='y')


    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax2.set_ylabel('Tokens per Second', color=color)
    sns.lineplot(data=df, x='n_gpu_layers', y='tokens_per_second', marker='o', ax=ax2, color=color, label='Tokens per Second')
    ax2.tick_params(axis='y', labelcolor=color)

    # Combine legends
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper center', bbox_to_anchor=(0.5, 1), ncol=2)
    ax2.get_legend().remove()

    ax1.grid(True)
    ax2.grid(False)

    plt.title('Performance Analysis of GPU Layer Offloading: GPU RAM and Tokens/second')
    plt.tight_layout()
    plt.savefig(f'{results_dir}/graph_ngl.png')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--hardware', type=str, default='RTX_2080', help='Hardware to show results for')
    args = parser.parse_args()

    results_data = load_results(hardware=args.hardware)
    plot_results(results_data)
