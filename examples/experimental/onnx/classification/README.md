# Classification sample

This sample shows an example of quantization of classification models. The used dataset is ImageNet.

## Install

Please, install the requirements for ONNX Post-Training Quantization of NNCF.

Install requirements

```
pip install -r <nncf dir>/nncf/experimental/onnx/requirements.txt
pip install -r <nncf dir>/examples/experimental/onnx/requirements.txt
```

## Getting the quantized model

To run post-training quantization on your model you can use the following command.

```
python onnx_ptq_classification.py -m <ONNX model path> -o <quantized ONNX model path> --data <ImageNet data path>
```

Also, you could specify some options of quantization, please, take a look at the argument description by using the command:

```
python onnx_ptq_classification.py --help
```

## Measuring the accuracy of the original and quantized models

If you would like to compare the accuracy of the original model and quantized one, you could
use [accuracy_checker](https://github.com/openvinotoolkit/open_model_zoo/tree/master/tools/accuracy_checker). The
necessary config files are located [here](./examples/experimental/onnx/classification/ac_configs/). The thing that you only need is to
fill in the config with the following infromation: the path to ImageNet folder and the path to the annotation file. The
accuracy checker config for the original and quantized models is the same.

Use the following command to get the model accuracy:

```
accuracy_check -c <path to config fileh> -m <ONNX model>
```

## Results of Post-Training quantization of ONNX models

|           Model           | Original accuracy | Quantized model accuracy |
|:-------------------------:|:-----------------:|:------------------------:|
|         ResNet-50         |      75.17%       |          74.74%          |
|        MobilenetV2        |      71.87%       |          71.29%          |
| InceptionV1 (GoogleNetV1) |      69.77%       |          69.64%          |
| InceptionV3 (GoogleNetV3) |      77.45%       |          77.30%          |
|      SqueezenetV1.1       |      58.19%       |          57.72%          |

## Measuring the performance of the original and quantized models

If you would like to compare the performance of the original model and quantized one, you could
use [benchmark_tool](https://github.com/openvinotoolkit/openvino/tree/master/tools/benchmark_tool).

Use the following command to get the model performance numbers:

```
benchmark_app -m <ONNX model>
```