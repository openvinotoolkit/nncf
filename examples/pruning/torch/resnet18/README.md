# Pruning-Aware Training: An Example for Resnet18 in PyTorch

This example demonstrates how to prune a PyTorch ResNet-18 using NNCF`s Pruning API,
then recover accuracy through pruning-aware training, starting with a model initially pretrained on Tiny ImageNet-200.

The example includes the following steps:

- Loading the Tiny ImageNet-200 dataset (~237 Mb) and the Resnet18 PyTorch model pretrained on this dataset.
- Prune the model using NNCF algorithm by magnitude algorithm.
- Fine-tuning the pruned model for one epoch to improve its metrics.

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

It's pretty simple. The example does not require additional preparation. It will do the preparation itself, such as loading the dataset and model, etc.

```bash
# To run Magnitude-Based pruning
python main.py

# To run Magnitude-Based pruning with batch norm adaptation
python main.py --mode bn_adaptation

# To to run Regularization-Based pruning
python main.py --mode rb
```
