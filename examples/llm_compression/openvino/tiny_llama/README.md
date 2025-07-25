# Large Language Models Weight Compression Example

This example demonstrates how to optimize Large Language Models (LLMs) using NNCF weight compression API. The example applies 4/8-bit mixed-precision quantization to weights of Linear (Fully-connected) layers of [TinyLlama/TinyLlama-1.1B-Chat-v1.0](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0) model. This leads to a significant decrease in model footprint and performance improvement with OpenVINO.

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

To run the example:

```bash
python main.py
```

This will automatically:

- Download the TinyLlama model and dataset
- Apply weight compression using NNCF
- Save the optimized model
