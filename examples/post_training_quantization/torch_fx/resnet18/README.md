# Post-Training Quantization of Resnet18 PyTorch Model exported to torch.fx.GraphModule

This example demonstrates how to use Post-Training Quantization API from Neural Network Compression Framework (NNCF) to quantize PyTorch models exported to torch.fx.GraphModule on the example of Resnet18 post-training quantization, pretrained on Tiny ImageNet-200 dataset.

The example includes the following steps:

- Loading the Tiny ImageNet-200 dataset (~237 Mb) and the Resnet18 PyTorch model pretrained on this dataset.
- Exporting model to torch.fx.GraphModule by torch.export.export function.
- Quantizing the model using NNCF Post-Training Quantization algorithm.
- Output of the following characteristics of the quantized model:
  - Accuracy drop of the quantized model (INT8) over the pre-trained model (FP32)
  - Performance speed up of the quantized model (INT8)

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
