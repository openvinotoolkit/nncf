# Post-Training Quantization of ONNX model

This example demonstrates an experimental feature of NNCF - post-training quantization of the ONNX model.
The algorithm takes the original ONNX model, makes the network analysis, seeking the model's graph insertion points for quantizers.
Then it places extra outputs in the model's graph to collect tensor statistics for quantizer initialization.
To collect statistics or infer the ONNX model ONNXRuntime with OpenVINO Execution Provider is used. 
As the last step, the initialized quantizers are added to the ONNX model's graph.

This feature was tested on torchvision ResNet-50 model exported with default parameters of export. Only ImageNet dataset is currently supported.

## Run Post-Training quantization sample
```
python post_training_quantization.py -m <ONNX model path> -o <quantized ONNX model path> --data <Imagenet data path>
```

## Results of Post-Training quantization of ONNX model

|Model|Original accuracy|Quantized model accuracy|
| :---: | :---: | :---: |
|ResNet-50|76.13%|75.44%|