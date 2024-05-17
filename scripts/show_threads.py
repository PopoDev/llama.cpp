import os
import json
import matplotlib.pyplot as plt
import seaborn as sns

def load_results(results_dir='results/threads'):
    data = []
    for filename in os.listdir(results_dir):
        if filename.endswith('.json'):
            with open(os.path.join(results_dir, filename), 'r') as f:
                data.append(json.load(f))
    return data

def plot_results(data):
    results_dir = 'results'
    n_threads = [result['n_threads'] for result in data]
    tokens_per_second = [result['tokens_per_second'] for result in data]

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=n_threads, y=tokens_per_second, marker='o')
    plt.title('Performance Analysis of Multi-threading on Tokens/second')
    plt.xlabel('Number of Threads')
    plt.ylabel('Tokens per Second')
    plt.xticks(n_threads)
    plt.tight_layout()

    plt.savefig(f'{results_dir}/graph_threads.png')

if __name__ == "__main__":
    results_data = load_results()
    plot_results(results_data)
