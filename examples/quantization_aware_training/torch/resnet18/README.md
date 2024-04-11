# Quantization-Aware Training: An Example for Resnet18 in PyTorch

This example demonstrates how to use Post-Training Quantization API from Neural Network Compression Framework (NNCF) to quantize and train PyTorch models on the example of Resnet18 quantization aware training, pretrained on Tiny ImageNet-200 dataset.

The example includes the following steps:

- Loading the Tiny ImageNet-200 dataset (~237 Mb) and the Resnet18 PyTorch model pretrained on this dataset.
- Quantizing the model using NNCF Post-Training Quantization algorithm.
- Fine tuning quantized model for one epoch to improve quantized model metrics.
- Output of the following characteristics of the quantized model:
  - Accuracy drop of the quantized model (INT8) over the pre-trained model (FP32)
  - Compression rate of the quantized model file size relative to the pre-trained model file size
  - Performance speed up of the quantized model (INT8)

## Install requirements

At this point it is assumed that you have already installed NNCF. You can find information on installation NNCF [here](https://github.com/openvinotoolkit/nncf#user-content-installation).

To work with the example you should install the corresponding Python package dependencies:

```bash
pip install -r requirements.txt
```

## Run Example

It's pretty simple. The example does not require additional preparation. It will do the preparation itself, such as loading the dataset and model, etc.

```bash
python main.py
```
