# Quantization-Aware Training of STFPM PyTorch model from Anomalib

The anomaly detection domain is one of the domains in which models are used in scenarios where the cost of model error is high and accuracy cannot be sacrificed for better model performance. Quantization-Aware Training (QAT) is perfect for such cases, as it reduces quantization error without model performance degradation by training the model.

This example demonstrates how to quantize [Student-Teacher Feature Pyramid Matching (STFPM)](https://anomalib.readthedocs.io/en/latest/markdown/guides/reference/models/image/stfpm.html) PyTorch model from [Anomalib](https://github.com/openvinotoolkit/anomalib) using Quantization API from Neural Network Compression Framework (NNCF). At the first step, the model is quantized using Post-Training Quantization (PTQ) algorithm to obtain the best initialization of the quantized model. If the accuracy of the quantized model after PTQ does not meet requirements, the next step is to train the quantized model using PyTorch framework.

NNCF provides a seamless transition from Post-Training Quantization to Quantization-Aware Training without additional model preparation and transfer of magic parameters.

The example includes the following steps:

- Loading the [MVTec (capsule category)](https://www.mvtec.com/company/research/datasets/mvtec-ad) dataset (~4.9 Gb).
- (Optional) Training STFPM PyTorch model from scratch.
- Loading STFPM model pretrained on this dataset.
- Quantizing the model using NNCF Post-Training Quantization algorithm.
- Fine-tuning quantized model for one epoch to improve quantized model metrics.
- Output of the following characteristics of the quantized model:
  - Accuracy drop of the quantized model (INT8) over the pre-trained model (FP32)
  - Compression rate of the quantized model file size relative to the pre-trained model file size
  - Performance speed up of the quantized model (INT8)

## Install requirements

At this point, it is assumed that you have already installed NNCF. You can find information on installation of NNCF [here](https://github.com/openvinotoolkit/nncf#user-content-installation).

To work with the example you should install the corresponding Python package dependencies:

```bash
pip install -r requirements.txt
```

## Run Example

It's pretty simple. The example does not require additional preparation. It will do the preparation itself, such as loading the dataset and model, etc.

```bash
python main.py
```
