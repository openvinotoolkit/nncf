
# Classification sample

This sample shows an example of quantization of classification models from torchvision. 
The used dataset is ImageNet.


## Install 

To correctly use the sample you should follow the instructions below.

Install requirements

```
pip install -r <nncf dir>/nncf/experimental/onnx/requirements.txt
```

## Run Post-Training quantization sample

Take a notice that the ONNX model opset should be equal 13.

```
python post_training_quantization.py -m <ONNX model path> -o <quantized ONNX model path> --data <ImageNet data path>
```

## Run validation

You could use [accuracy_checker](https://github.com/openvinotoolkit/open_model_zoo/tree/master/tools/accuracy_checker) to run the validation of the obtained model. The config file is the same for resnet50 and mobilenetv2 and it is located [here](/.ac_configs).  


## Results of Post-Training quantization of ONNX model

|Model|Original accuracy|Quantized model accuracy|
| :---: | :---: | :---: |
|ResNet-50|76.13%|75.97%|
|MobilenetV2|71.87%|71.35%|