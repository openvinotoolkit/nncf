# GenAI Optimizations

This module provides experimental optimizations for GenAI models in PyTorch. The goal is to improve efficiency and performance for generative AI tasks while minimizing accuracy loss.

## Supported Generative AI Scenarios

- Multimodal understanding and text generation

## Supported Generative AI Optimization Methods

- [**Visual Token Pruning**](./visual_token_pruning.py):
  Designed to accelerate inference in VLMs, where the number of input visual tokens is often significantly larger than that of textual tokens. Pruning these tokens reduces first-token latency and overall FLOPs while preserving accuracy. In this repository, we implement a visual token pruning method called [CDPruner](https://arxiv.org/pdf/2506.10967), which maximizes the conditional diversity of retained tokens. It can reduce FLOPs by 95% and CUDA latency by 78%, while maintaining 94% of the original accuracy.

## Benchmarks

- [MME: A Comprehensive Evaluation Benchmark for Multimodal Large Language Models](./benchmarks/mmebench.py). Supported and tested models include:
  - [llava-hf/llava-1.5-7b-hf](https://huggingface.co/llava-hf/llava-1.5-7b-hf)
  - [llava-hf/llava-v1.6-mistral-7b-hf](https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf)
  - [Qwen/Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)
  - [Qwen/Qwen2-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct)
