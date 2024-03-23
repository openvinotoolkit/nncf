# Large Language Models Weight Compression Example

This example demonstrates how to optimize Large Language Models (LLMs) using NNCF weight compression API. The example applies 4/8-bit mixed-precision quantization to weights of Linear (Fully-connected) layers of [TinyLlama/TinyLlama-1.1B-Chat-v1.0](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0) model. This leads to a significant decrease in model footprint and performance improvement with OpenVINO.

## Prerequisites

To use this example:

- Create a separate Python* environment and activate it: `python3 -m venv nncf_env && source nncf_env/bin/activate`
- Install dependencies:

```bash
pip install -U pip
pip install -r requirements.txt
pip install ../../../../
```

## Run Example

To run example:

```bash
python main.py
```

It will automatically download the dataset and baseline model and save the resulting model.
