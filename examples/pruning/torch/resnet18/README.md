# Quantization-Aware Training: An Example for Resnet18 in PyTorch

This example demonstrates how to use Pruning API from Neural Network Compression Framework (NNCF) to prune
and train PyTorch models on the example of Resnet18 prune aware training, pretrained on Tiny ImageNet-200 dataset.

The example includes the following steps:

- Loading the Tiny ImageNet-200 dataset (~237 Mb) and the Resnet18 PyTorch model pretrained on this dataset.
- Prune the model using NNCF algorithm by magnitude algorithm.
- Fine tuning quantized model for one epoch to improve quantized model metrics.

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

It's pretty simple. The example does not require additional preparation. It will do the preparation itself, such as loading the dataset and model, etc.

```bash
python main.py
```
