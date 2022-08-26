# Classification sample

This sample shows an example of quantization of classification models: 
ResNet-50, MobilenetV2, InceptionV1 (GoogleNetV1), InceptionV3 (GoogleNetV3), SqueezenetV1.1. The used dataset is ImageNet.

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


## Results of Post-Training quantization of ONNX models

|           Model           | Original accuracy | Quantized model accuracy |
|:-------------------------:|:-----------------:|:------------------------:|
|         ResNet-50         |      75.17%       |          74.74%          |
|        MobilenetV2        |      71.87%       |          71.29%          |
| InceptionV1 (GoogleNetV1) |      69.77%       |          69.64%          |
| InceptionV3 (GoogleNetV3) |      77.45%       |          77.30%          |
|      SqueezenetV1.1       |      58.19%       |          57.72%          |
