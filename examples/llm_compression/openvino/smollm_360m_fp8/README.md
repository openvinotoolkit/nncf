# Large Language Models FP8 Compression Example

This example demonstrates how to optimize Large Language Models (LLMs) using NNCF quantize API. The example applies FP8 quantization to [HuggingFaceTB/SmolLM-360M](https://huggingface.co/HuggingFaceTB/SmolLM-360M) model. This leads to a significant decrease in model footprint and performance improvement with OpenVINO.

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
