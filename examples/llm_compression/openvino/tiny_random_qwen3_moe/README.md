# Qwen3.5-MoE Data-Aware Weight Compression Example

This example demonstrates how to apply data-aware weight compression to [optimum-intel-internal-testing/tiny-random-qwen3.5-moe](https://huggingface.co/optimum-intel-internal-testing/tiny-random-qwen3.5-moe) model.

## Prerequisites

Before running this example, ensure you have Python 3.10+ installed and set up your environment:

### 1. Create and activate a virtual environment

```bash
python3 -m venv nncf_env
source nncf_env/bin/activate  # On Windows: nncf_env\Scripts\activate.bat
```

### 2. Install NNCF and other dependencies

```bash
python3 -m pip install ../../../../ -r requirements.txt
python3 -m pip install transformers==5.2.0  # Required for Qwen3.5-MoE
```

## Run Example

To run example:

```bash
python main.py
```

This will automatically:

- Download the Qwen3.5-MoE model
- Apply data-aware weight compression with scale estimation and AWQ using NNCF
- Save the optimized model
