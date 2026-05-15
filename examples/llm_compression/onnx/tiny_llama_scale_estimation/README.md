# Compress TinyLLama model using data

This example demonstrates how to optimize Large Language Models (LLMs) using NNCF weight compression API & data for the advanced algorithms usage. The example applies 4/8-bit mixed-precision quantization & Scale Estimation algorithm to weights of Linear (Fully-connected) layers of [TinyLlama/TinyLlama-1.1B-Chat-v1.0](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0) model. This leads to a significant decrease in model footprint and performance improvement with OpenVINO Runtime.

The example includes the following steps:

- Prepare `TinyLlama/TinyLlama-1.1B-Chat-v1.0` text-generation model in ONNX format.
- Prepare `wikitext-2-raw-v1` dataset.
- Compress weights of the model with NNCF Weight compression algorithm with Scale Estimation & dataset.

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
```

## Run Example

To run example:

```bash
python main.py
```

This will automatically:

- Download the TinyLlama model and dataset
- Apply weight compression using NNCF
- Save the optimized model
