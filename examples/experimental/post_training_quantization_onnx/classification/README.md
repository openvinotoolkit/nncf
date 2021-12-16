
# Classification sample

This sample shows an example of quantization of classification models from torchvision. 
The used dataset is ImageNet.

## Run Post-Training quantization sample

Take a notice that the ONNX model opset should be equal 13.

```
python post_training_quantization.py -m <ONNX model path> -o <quantized ONNX model path> --data <ImageNet data path>
```

## Results of Post-Training quantization of ONNX model

|Model|Original accuracy|Quantized model accuracy|
| :---: | :---: | :---: |
|ResNet-50|76.13%|75.97%|
|MobilenetV2|71.87%|71.35%|