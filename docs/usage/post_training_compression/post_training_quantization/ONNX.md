# Post-Training Quantization for ONNX

NNCF supports [ONNX](https://onnx.ai/) backend for the Post-Training Quantization algorithm.
This guide contains some notes that you should consider before working with NNCF for ONNX.

## Model Preparation

The majority of the ONNX models are exported from different frameworks, such as PyTorch or TensorFlow.

NNCF fully supports ONNX models with the opset 13 or higher. \
NNCF supports only per-tensor quantization for ONNX models with the opset 10, 11, and 12. \
NNCF does not support ONNX models with the opset lower than 10.

If you have an ONNX model with the opset lower than 13, we recommend to update the model to a higher opset version. \
If you obtained an ONNX model from other frameworks, you can try to update an export function by setting a higher target opset version. \
If you do not have access to the export function, you can use the converter function from the ONNX package. See the example below.

```python
import onnx
from onnx.version_converter import convert_version

model = onnx.load_model('/path_to_model')
converted_model = convert_version(model, target_version=13)
```

## ONNX Results

Below are some results obtained using [benchmarking section](/tests/onnx/benchmarking/README.md) for the models from [ONNX Model Zoo](https://github.com/onnx/models).

### Classification

|     ONNX Model      |Compression algorithm|Dataset|Accuracy (Drop) %|
|:-------------------:| :---: | :---: | :---: |
|    [ResNet-50](https://github.com/onnx/models/tree/8e893eb39b131f6d3970be6ebd525327d3df34ea/vision/classification/resnet/model/resnet50-v2-7.onnx)    |INT8 (Post-Training)|ImageNet|74.63 (0.21)|
|   [ShuffleNet](https://github.com/onnx/models/tree/8e893eb39b131f6d3970be6ebd525327d3df34ea/vision/classification/shufflenet/model/shufflenet-9.onnx)    |INT8 (Post-Training)|ImageNet|47.25 (0.18)|
|    [GoogleNet](https://github.com/onnx/models/tree/8e893eb39b131f6d3970be6ebd525327d3df34ea/vision/classification/inception_and_googlenet/googlenet/model/googlenet-12.onnx)    |INT8 (Post-Training)|ImageNet|66.36 (0.3)|
| [SqueezeNet V1.0](https://github.com/onnx/models/tree/8e893eb39b131f6d3970be6ebd525327d3df34ea/vision/classification/squeezenet/model/squeezenet1.0-12.onnx) |INT8 (Post-Training)|ImageNet|54.3 (0.54)|
|  [MobileNet V2](https://github.com/onnx/models/tree/8e893eb39b131f6d3970be6ebd525327d3df34ea/vision/classification/mobilenet/model/mobilenetv2-12.onnx)   |INT8 (Post-Training)|ImageNet|71.38 (0.49)|
|  [DenseNet-121](https://github.com/onnx/models/tree/8e893eb39b131f6d3970be6ebd525327d3df34ea/vision/classification/densenet-121/model/densenet-12.onnx)   |INT8 (Post-Training)|ImageNet|60.16 (0.8)|
|     [VGG-16](https://github.com/onnx/models/tree/8e893eb39b131f6d3970be6ebd525327d3df34ea/vision/classification/vgg/model/vgg16-12.onnx)      |INT8 (Post-Training)|ImageNet|72.02 (0.0)|

### Object Detection

|   ONNX Model    |Compression algorithm| Dataset |mAP (drop) %|
|:---------------:| :---: | :---: | :---: |
|   [SSD1200](https://github.com/onnx/models/tree/8e893eb39b131f6d3970be6ebd525327d3df34ea/vision/object_detection_segmentation/ssd/model/ssd-12.onnx)   |INT8 (Post-Training)|COCO2017|20.17 (0.17)|
| [Tiny-YOLOv2](https://github.com/onnx/models/tree/8e893eb39b131f6d3970be6ebd525327d3df34ea/vision/object_detection_segmentation/tiny-yolov2/model/tinyyolov2-8.onnx) |INT8 (Post-Training)|VOC12|29.03 (0.23)|
