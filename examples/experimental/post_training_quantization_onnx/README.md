# Post-Training Quantization of ONNX model

This examples demonstrate an experimental feature of NNCF - post-training quantization of the ONNX model.
The algorithm takes the original ONNX model, makes the network analysis, seeking the model's graph insertion points for quantizers.
Then it places extra outputs in the model's graph to collect tensor statistics for quantizer initialization.
To collect statistics or infer the ONNX model ONNXRuntime with OpenVINO Execution Provider is used. 
As the last step, the initialized quantizers are added to the ONNX model's graph.

This feature was tested on torchvision ResNet-50 and MobilenetV2 models exported with default parameters of export.

## Results of Post-Training quantization of ONNX model

|Model|Original accuracy|Quantized model accuracy|
| :---: | :---: | :---: |
|ResNet-50|76.13%|75.97%|
|MobilenetV2|71.87%|71.35%|
|Yolov5s|37.1%|36.1%|