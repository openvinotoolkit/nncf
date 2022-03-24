
# Classification sample

This sample shows an example of quantization of classification models. 
The used dataset is ImageNet.


## Install 

To correctly use the sample you should follow the instructions below.

Install requirements

```
pip install -r <nncf dir>/nncf/experimental/onnx/requirements.txt
```

## Run Post-Training quantization sample

```
python -m <ONNX model path> -o <quantized ONNX model path> --data <ImageNet data path>
```

## Run validation

You could use [accuracy_checker](https://github.com/openvinotoolkit/open_model_zoo/tree/master/tools/accuracy_checker) to run the validation of the obtained model. The AccuracyChecker config files are located [here](examples/experimental/onnx/ac_configs/).  


## Results of Post-Training quantization of ONNX models

|          Model           | Original accuracy | Quantized model accuracy |
|:------------------------:|:-----------------:|:------------------------:|
|        ResNet-50         |      75.17%       |          74.74%          |
|       MobilenetV2        |      71.87%       |          71.33%          |
| InceptionV1(googlenetV1) |      69.77%       |          69.72%          |
|       InceptionV3        |      77.45%       |          77.31%          |
|      SqueezenetV1.1      |      58.19%       |          57.73%          |