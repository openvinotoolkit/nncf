# Large Language Models FP8 Compression Example

This example demonstrates how to apply static FP8 quantization to [HuggingFaceTB/SmolLM2-360M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct) model. It can be useful for evaluation and early HW enablement purposes.

## Prerequisites

Before running this example, ensure you have Python 3.9+ installed and set up your environment:

### 1. Create and activate a virtual environment

```bash
python3 -m venv nncf_env
source nncf_env/bin/activate  # On Windows: nncf_env\Scripts\activate.bat
```

### 2. Install NNCF and other dependencies

```bash
python3 -m pip install ../../../../ -r requirements.txt
```

## Run Example

To run example:

```bash
python main.py
```

This will automatically:

- Download the SmolLM2 model and dataset
- Apply weight compression using NNCF
- Save the optimized model
