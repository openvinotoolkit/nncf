# Qantization-aware Training after Post-training Quantization Suite

This is the test suite based on QAT examples training and validation code that takes all samples quantization configs and applies PTQ to the correspondent models. It compares metrics between original and quantized models and tries to recover metrics by QAT.

## Installation

```bash
make install-torch-test
```

## Usage

Once the environment is installed use the following command to run all tests:

```bash
python -m pytest tests/torch/qat --sota-data-dir /path/to/omz/training/datasets --sota-checkpoints-dir /path/to/compression_training_baselines
```
