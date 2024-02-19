# Qantization-aware Training after Post-training Quantization Suite

This is the test suite based on QAT examples training and validation code that takes all samples quantization configs and applies PTQ to the correspondent models. It compares metrics between original and quantized models and tries to recover metrics by QAT.

## Installation

For the Torch backend:

```bash
make install-torch-test
```

## Usage

Once the environment is installed use the following command to run all tests for the Torch backend:

```bash
python -m pytest tests/qat_after_ptq/torch --data /path/to/omz/training/datasets --weights /path/to/nncf/checkpoints
```
