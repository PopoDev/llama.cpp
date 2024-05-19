# llama.cpp

Inference of Meta's [LLaMA](https://arxiv.org/abs/2302.13971) model (and others) in pure C/C++

## Description

The main goal of `llama.cpp` is to enable LLM inference with minimal setup and state-of-the-art performance.

- Plain C/C++ implementation without any dependencies
- AVX, AVX2 and AVX512 support for x86 architectures
- 1.5-bit, 2-bit, 3-bit, 4-bit, 5-bit, 6-bit, and 8-bit integer quantization for faster inference and reduced memory use
- Custom CUDA kernels for running LLMs on NVIDIA GPUs
- CPU+GPU hybrid inference to partially accelerate models larger than the total VRAM capacity

## Build

Use make to build the project:

```bash
make
```

### CUDA

This provides GPU acceleration using the CUDA cores of your Nvidia GPU. Make sure to have the CUDA toolkit installed. You can download it from your Linux distro's package manager (e.g. `apt install nvidia-cuda-toolkit`) or from here: [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads).

Make sure that the the `nvcc` compiler is in your `PATH`.

```bash
export PATH=/usr/local/cuda/bin:$PATH
```

Use the flag `LLAMA_CUDA=1` to enable CUDA support:

```bash
LLAMA_CUDA=1 make
```

## Usage

### Download a model
After building the project, you need to download a model in GGUF format. For example we can download the `Mistral-7B-Instruct-v0.2-GGUF` model from [TheBloke](https://huggingface.co/TheBloke?sort_models=downloads#models).

```bash
# Download the 4-bit quantized model
wget https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf
```

### Run the model

After downloading the model, you can run inference using the following command:

```bash
./main -ngl 35 -m mistral-7b-instruct-v0.2.Q4_K_M.gguf --color -c 32768 --temp 0.7 --repeat_penalty 1.1 -n -1  -p "Who won the 2018 world cup?"
```

## Experiments

### GPU acceleration

We run experiments varying the `n_gpu_layers` parameter to see how the GPU acceleration affects the inference time. We use the `Mistral-7B-Instruct-v0.2-GGUF` model on a NVIDIA RTX 2080 GPU and a context size of 32768. The prompt is "What is the capital of the United States?". We use the seed `470` for reproducibility.

```bash
# Quick run:
./main -ngl 35 -c 32768 -s 470 -m mistral-7b-instruct-v0.2.Q4_K_M.gguf -p "What is the capital of the United States?"

# Script to run the ngl experiment
python scripts/run_ngl.py

# Script to plot the results
python scripts/show_ngl.py
```

### Multi-threading

We run experiments varying the number of threads to see how multi-threading affects the inference time. We use the `Mistral-7B-Instruct-v0.2-GGUF` model on a Intel i7-8700K CPU and a context size of 32768. The prompt is "What is the capital of the United States?". We use the seed `470` for reproducibility.

```bash
# Quick run:
./main -t 1 -c 32768 -s 470 -m mistral-7b-instruct-v0.2.Q4_K_M.gguf -p "What is the capital of the United States?"

# Script to run the threads experiment
python scripts/threads.py

# Script to plot the results
python scripts/show_threads.py
```

### CPU-only vs GPU acceleration

- CPU: Intel i7-5820K CPU @ 3.30GHz with 6 cores and 2 threads per core
- GPU: NVIDIA RTX 2080 with 12GB of VRAM

```bash
# CPU-only
./main -t -c 32768 -s 470 -m mistral-7b-instruct-v0.2.Q4_K_M.gguf -p "What is the capital of the United States?"

# GPU acceleration
./main -ngl 35 -c 32768 -s 470 -m mistral-7b-instruct-v0.2.Q4_K_M.gguf -p "What is the capital of the United States?"
```

## Results
- CPU: 8.47 tokens/s
- GPU: 81.89 tokens/s

