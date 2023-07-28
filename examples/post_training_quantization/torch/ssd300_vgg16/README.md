# Post-Training Quantization of SSD PyTorch Model

This example demonstrates how to use Post-Training Quantization API from Neural Network Compression Framework (NNCF) to quantize PyTorch models on the example of [SSD300_VGG16](https://pytorch.org/vision/main/models/generated/torchvision.models.detection.ssd300_vgg16.html) from torchvision library.

The example includes the following steps:

- Loading the [COCO128](https://www.kaggle.com/datasets/ultralytics/coco128) dataset (~7 Mb).
- Loading [SSD300_VGG16](https://pytorch.org/vision/main/models/generated/torchvision.models.detection.ssd300_vgg16.html) from torchvision pretrained on the full COCO dataset.
- Patching some internal methods with `no_nncf_trace` context so that the model graph is traced properly by NNCF.
- Quantizing the model using NNCF Post-Training Quantization algorithm.
- Output of the following characteristics of the quantized model:
  - Accuracy drop of the quantized model (INT8) over the pre-trained model (FP32).
  - Compression rate of the quantized model file size relative to the pre-trained model file size.
  - Performance speed up of the quantized model (INT8).

## Install requirements

At this point it is assumed that you have already installed NNCF. You can find information on installation NNCF [here](https://github.com/openvinotoolkit/nncf#user-content-installation).

To work with the example you should install the corresponding Python package dependencies:

```bash
pip install -r requirements.txt
```

## Run Example

The example does not require any additional preparation, just run:

```bash
python main.py
```
