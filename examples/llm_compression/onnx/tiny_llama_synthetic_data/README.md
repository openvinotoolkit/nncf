# Compress TinyLLama model using synthetic data

This example demonstrates how to optimize Large Language Models (LLMs) using NNCF weight compression API & synthetic data for the advanced algorithms usage. The example applies 4/8-bit mixed-precision quantization & Scale Estimation algorithm to weights of Linear (Fully-connected) layers of [TinyLlama/TinyLlama-1.1B-Chat-v1.0](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0) model. This leads to a significant decrease in model footprint and performance improvement with OpenVINO Runtime.

The example includes the following steps:

- Prepare `TinyLlama/TinyLlama-1.1B-Chat-v1.0` text-generation model in ONNX format.
- Prepare `synthetic` dataset using `nncf.data.generate_text_data` method.
- Compress weights of the model with NNCF Weight compression algorithm with Scale Estimation & `synthetic` dataset.

## Install requirements

To use this example:

- Create a separate Python* environment and activate it: `python3 -m venv nncf_env && source nncf_env/bin/activate`
- Install dependencies:

```bash
pip install -U pip
pip install -r requirements.txt
pip install ../../../../
```

## Run Example

The example is fully automated. Just run the following command in the prepared Python environment:

```bash
python main.py
```
