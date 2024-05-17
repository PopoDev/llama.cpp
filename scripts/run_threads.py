import subprocess
import os
import re
import json

MAX_THREADS = 16

def run_command(n_threads):
    cmd = f'./main -t {n_threads} -c 32768 -s 470 -m mistral-7b-instruct-v0.2.Q4_K_M.gguf -p "What is the capital of the United States?"'
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result

def save_results(n_threads, tokens_per_second, cpu_ram):
    results = {
        "n_threads": n_threads,
        "tokens_per_second": tokens_per_second,
        "cpu_ram_mib": cpu_ram,
    }
    if not os.path.exists('results/threads'):
        os.makedirs('results/threads')
    with open(f'results/threads/threads_{n_threads}.json', 'w') as f:
        json.dump(results, f, indent=4)

def parse_output(output):
    n_threads_pattern = r'n_threads\s*=\s*(\d+)\s*/\s*(\d+)'
    eval_time_pattern = r'eval time\s*=\s*[\d.]+\s*ms\s*/\s*[\d.]+\s*runs\s*\(\s*[\d.]+\s*ms\s*per\s*token,\s*([\d.]+)\s*tokens\s*per\s*second\)'
    cpu_ram_pattern = r'llama_print_ram: CPU RAM \(MiB\) = (\d+)'

    n_threads_match = re.search(n_threads_pattern, output)
    eval_time_match = re.search(eval_time_pattern, output)
    cpu_ram_match = re.search(cpu_ram_pattern, output)

    n_threads = 0
    total_threads = 0
    tokens_per_second = 0
    cpu_ram = 0

    if n_threads_match:
        n_threads = int(n_threads_match.group(1))
        total_threads = int(n_threads_match.group(2))
        
    if eval_time_match:
        tokens_per_second = float(eval_time_match.group(1))

    if cpu_ram_match:
        cpu_ram = int(cpu_ram_match.group(1))

    return n_threads, total_threads, tokens_per_second, cpu_ram

if __name__ == "__main__":
    n_threads = MAX_THREADS

    while n_threads >= 1:
        result = run_command(n_threads)
        print(f"Trying with -t {n_threads}:")
        print(result.stderr)

        n_threads, total_threads, tokens_per_second, cpu_ram = parse_output(result.stderr)

        if total_threads > MAX_THREADS:
            print(f"MAX_THREADS={MAX_THREADS} is not covering all threads for this architecture.")
        
        save_results(n_threads, tokens_per_second, cpu_ram)
        n_threads -= 1
