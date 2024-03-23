# Post-Training Quantization of Anomaly Classification OpenVINO model with control of accuracy metric

The anomaly detection domain is one of the domains in which models are used in scenarios where the cost of model error is high and accuracy cannot be sacrificed for better model performance. For such cases, it is necessary to use quantization methods with accuracy control where the maximum accuracy drop can be specified as an argument of the method.

This example demonstrates how to quantize [Student-Teacher Feature Pyramid Matching (STFPM)](https://huggingface.co/alexsu52/stfpm_mvtec_capsule) OpenVINO model from [Anomalib](https://github.com/openvinotoolkit/anomalib) using Post-Training Quantization with accuracy control API from Neural Network Compression Framework (NNCF).

The `nncf.quantize_with_accuracy_control()` method quantizes a model with a specified accuracy drop and the `max_drop` parameter is passed to specify the maximum absolute difference between the quantized and pre-trained model.

The example includes the following steps:

- Loading the [MVTec (capsule category)](https://www.mvtec.com/company/research/datasets/mvtec-ad) dataset (~385 Mb) and the [STFPM OpenVINO model](https://huggingface.co/alexsu52/stfpm_mvtec_capsule) pretrained on this dataset.
- Quantizing the model using NNCF Post-Training Quantization algorithm with accuracy control.
- Output of the following characteristics of the quantized model:
  - Accuracy drop between the quantized model (INT8) and the pre-trained model (FP32)
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

The maximum accuracy drop you can pass as a command line argument. F1 score is calculted in range [0,1] for STFPM. Thus if you want to specify the maximum accuracy drop between the quantized and pre-trained model of 0.5% you must specify 0.005 as a command line argument:

```bash
python main.py 0.005
```
