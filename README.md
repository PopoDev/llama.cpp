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