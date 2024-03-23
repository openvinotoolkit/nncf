# Post-Training Quantization of MobileNet v2 PyTorch Model

This example demonstrates how to use Post-Training Quantization API from Neural Network Compression Framework (NNCF) to quantize PyTorch models on the example of [MobileNet v2](https://huggingface.co/alexsu52/mobilenet_v2_imagenette) quantization, pretrained on [Imagenette](https://github.com/fastai/imagenette) dataset.

The example includes the following steps:

- Loading the [Imagenette](https://github.com/fastai/imagenette) dataset (~340 Mb) and the [MobileNet v2 PyTorch model](https://huggingface.co/alexsu52/mobilenet_v2_imagenette) pretrained on this dataset.
- Quantizing the model using NNCF Post-Training Quantization algorithm.
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
