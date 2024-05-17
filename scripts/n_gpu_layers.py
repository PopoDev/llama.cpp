import subprocess
import os
import re
import json

MAX_GPU_LAYERS = 100

def run_command(n_gpu_layers):
    cmd = f'./main -ngl {n_gpu_layers} -c 32768 -s 470 -m mistral-7b-instruct-v0.2.Q4_K_M.gguf -p "What is the capital of the United States?"'
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result

def save_results(n_gpu_layers, tokens_per_second, cpu_ram, gpu_ram):
    results = {
        "n_gpu_layers": n_gpu_layers,
        "tokens_per_second": tokens_per_second,
        "cpu_ram_mib": cpu_ram,
        "gpu_ram_mib": gpu_ram
    }
    if not os.path.exists('results'):
        os.makedirs('results')
    with open(f'results/ngl_{n_gpu_layers}.json', 'w') as f:
        json.dump(results, f, indent=4)

def parse_output(output):
    offloaded_pattern = r'llm_load_tensors: offloaded (\d+)/(\d+) layers to GPU'
    eval_time_pattern = r'eval time\s*=\s*[\d.]+\s*ms\s*/\s*[\d.]+\s*runs\s*\(\s*[\d.]+\s*ms\s*per\s*token,\s*([\d.]+)\s*tokens\s*per\s*second\)'
    cpu_ram_pattern = r'llama_print_ram: CPU RAM \(MiB\) = (\d+)'
    gpu_ram_pattern = r'llama_print_ram: GPU RAM \(MiB\) = (\d+)'

    offloaded_match = re.search(offloaded_pattern, output)
    eval_time_match = re.search(eval_time_pattern, output)
    cpu_ram_match = re.search(cpu_ram_pattern, output)
    gpu_ram_match = re.search(gpu_ram_pattern, output)

    offloaded_layers = 0
    total_layers = 0
    tokens_per_second = 0
    cpu_ram = 0
    gpu_ram = 0

    if offloaded_match:
        offloaded_layers = int(offloaded_match.group(1))
        total_layers = int(offloaded_match.group(2))
    
    if eval_time_match:
        tokens_per_second = float(eval_time_match.group(1))

    if cpu_ram_match:
        cpu_ram = int(cpu_ram_match.group(1))

    if gpu_ram_match:
        gpu_ram = int(gpu_ram_match.group(1))

    return offloaded_layers, total_layers, tokens_per_second, cpu_ram, gpu_ram

if __name__ == "__main__":
    n_gpu_layers = MAX_GPU_LAYERS

    while n_gpu_layers >= 0:
        result = run_command(n_gpu_layers)
        print(f"Trying with -ngl {n_gpu_layers}:")
        print(result.stderr)

        offloaded_layers, total_layers, tokens_per_second, cpu_ram, gpu_ram = parse_output(result.stderr)

        if total_layers > MAX_GPU_LAYERS:
            print(f"MAX_GPU_LAYERS={MAX_GPU_LAYERS} is not covering all layers for this model.")

        if offloaded_layers == total_layers:
            n_gpu_layers = offloaded_layers
        
        save_results(n_gpu_layers, tokens_per_second, cpu_ram, gpu_ram)
        n_gpu_layers -= 1
