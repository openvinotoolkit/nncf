# Multimodal Large Language Models Optimization Example: MileBench

This example demonstrates how to optimize Multimodal Large Language Models (MLLMs) using an experimental visual token pruning algorithm. The example leverages [MileBench](https://arxiv.org/pdf/2404.18532), a pioneering benchmark designed to rigorously evaluate the multimodal long-context capabilities of MLLMs. MileBench encompasses diverse tasks requiring both comprehension and generation, and introduces two distinct evaluation sets— diagnostic and realistic — that systematically assess models’ capacity for long-context adaptation and effective task completion.

Visual token pruning enables significant acceleration of inference in VLMs, where the number of input visual tokens is often much larger than the number of textual tokens. By pruning these tokens, we reduce first-token latency and overall FLOPs while preserving accuracy.

## Prerequisites

Before running this example, ensure you have Python 3.10+ installed and set up your environment:

### 1. Create and activate a virtual environment

```bash
python3 -m venv nncf_env
source nncf_env/bin/activate  # On Windows: nncf_env\Scripts\activate.bat
```

### 2. Install NNCF and other dependencies

Install the NNCF package along with required dependencies:

```bash
python3 -m pip install \
    ../../../../ \
    -r ../../../../src/nncf/experimental/torch/genai_optimizations/requirements.txt \
    -r requirements.txt
```

## Run Example

To run the example:

```bash
python benchmarks/milebench.py \
    --subset WikiVQA \
    --model Qwen/Qwen2-VL-2B-Instruct \
    --enable_visual_pruning \
    --num_keep_tokens 64 \
    --theta 0.5
```

This will automatically:

- Download the selected model and dataset
- Apply the visual token pruning algorithm
- Evaluate the model and report the score
