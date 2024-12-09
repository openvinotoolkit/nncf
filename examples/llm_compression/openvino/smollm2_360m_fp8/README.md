# Large Language Models FP8 Compression Example

This example demonstrates how to apply static FP8 quantization to [HuggingFaceTB/SmolLM2-360M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct) model. It can be useful for evaluation and early HW enablement purposes.

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
