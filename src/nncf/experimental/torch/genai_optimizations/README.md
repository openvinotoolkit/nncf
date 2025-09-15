# GenAI Optimizations

This module provides experimental optimizations for GenAI models in PyTorch. The goal is to improve efficiency and performance for generative AI tasks while minimizing accuracy loss.

## Supported Generative AI Scenarios

- Visual language text generation

## Supported Generative AI Optimization Methods

- [**Visual Token Pruning**](./visual_token_pruning.py):
  Designed to accelerate inference in VLMs, where the number of input visual tokens is often significantly larger than that of textual tokens. Pruning these tokens reduces first-token latency and overall FLOPs while preserving accuracy. In this repository, we implement a visual token pruning method called [CDPruner](https://arxiv.org/pdf/2506.10967), which maximizes the conditional diversity of retained tokens. It can reduce FLOPs by 95% and CUDA latency by 78%, while maintaining 94% of the original accuracy.

## Supported and tested models

Multimodal Large Language Models:

- [llava-hf/llava-1.5-7b-hf](https://huggingface.co/llava-hf/llava-1.5-7b-hf)
- [llava-hf/llava-v1.6-mistral-7b-hf](https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf)
- [Qwen/Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)
- [Qwen/Qwen2-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct)

## Prerequisites

Before running benchmarks, ensure you have **Python 3.9+** installed and set up your environment.

### 1. Create and activate a virtual environment

```bash
python3 -m venv nncf_env
source nncf_env/bin/activate      # On Windows: nncf_env\Scripts\activate.bat
```

### 2. Install NNCF and other dependencies

```bash
python3 -m pip install ../../../ -r requirements.txt```
```

## Benchmarks

### [MME: A Comprehensive Evaluation Benchmark for Multimodal Large Language Models](https://arxiv.org/pdf/2306.13394)

MME measures both perception and cognition abilities across 14 subtasks. Its concise instruction design enables fair comparison of MLLMs without the need for extensive prompt engineering.

Run the following command in the prepared Python environment:

```bash
python benchmarks/mmebench.py \
    --subset artwork \
    --model Qwen/Qwen2.5-VL-3B-Instruct \
    --enable_visual_pruning \
    --num_keep_tokens 128 \
    --theta 0.5
```

### [MILEBENCH: Benchmarking MLLMs in Long Context](https://arxiv.org/pdf/2404.18532)

MILEBENCH is a pioneering benchmark designed to rigorously evaluate the multimodal long-context capabilities of MLLMs. It encompasses diverse tasks requiring both comprehension and generation, and introduces two distinct evaluation sets—diagnostic and realistic—that systematically assess models’ capacity for long-context adaptation and effective task completion.

Run the following command in the prepared Python environment:

```bash
python benchmarks/milebench.py \
    --subset WikiVQA \
    --model Qwen/Qwen2-VL-2B-Instruct \
    --enable_visual_pruning \
    --num_keep_tokens 64 \
    --theta 0.5
```
