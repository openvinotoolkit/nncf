# Post-Training Quantization of MobileNet v2 ONNX Model
This example demonstrates how to use Post-Training Quantization API from Neural Network Compression Framework (NNCF) to quantize ONNX models on the example of [MobileNet v2](https://huggingface.co/alexsu52/mobilenet_v2_imagenette) quantization, pretrained on [Imagenette](https://github.com/fastai/imagenette) dataset.


The example includes the following steps:
- Loading the [Imagenette](https://github.com/fastai/imagenette) dataset (~340 Mb) and the [MobileNet v2 ONNX model](https://huggingface.co/alexsu52/mobilenet_v2_imagenette) pretrained on this dataset.
- Quantizing the model using NNCF Post-Training Quantization algorithm.
- Output of the following characteristics of the quantized model:
    - Accuracy drop of the quantized model (INT8) over the pre-trained model (FP32)
    - Performance speed up of the quantized model (INT8)

# Install requirements
At this point it is assumed that you have already installed NNCF. You can find information on installation NNCF [here](https://github.com/openvinotoolkit/nncf#user-content-installation).

To work with the example you should install the corresponding Python package dependencies:
```
pip install -r requirements.txt
```

# Run Example
It's pretty simple. The example does not require additional preparation. It will do the preparation itself, such as loading the dataset and model, etc.
```
python main.py
```


# ONNX Results

There are some results for the models from [ONNX Model Zoo](https://github.com/onnx/models)

### Classification

|     ONNX Model      |Compression algorithm|Dataset|Accuracy (Drop) %|
|:-------------------:| :---: | :---: | :---: |
|    [ResNet-50](https://github.com/onnx/models/tree/8e893eb39b131f6d3970be6ebd525327d3df34ea/vision/classification/resnet/model/resnet50-v2-7.onnx)    |INT8|ImageNet|74.63 (0.21)|
|   [ShuffleNet](https://github.com/onnx/models/tree/8e893eb39b131f6d3970be6ebd525327d3df34ea/vision/classification/shufflenet/model/shufflenet-9.onnx)    |INT8|ImageNet|47.25 (0.18)|
|    [GoogleNet](https://github.com/onnx/models/tree/8e893eb39b131f6d3970be6ebd525327d3df34ea/vision/classification/inception_and_googlenet/googlenet/model/googlenet-12.onnx)    |INT8|ImageNet|66.36 (0.3)|
| [SqueezeNet V1.0](https://github.com/onnx/models/tree/8e893eb39b131f6d3970be6ebd525327d3df34ea/vision/classification/squeezenet/model/squeezenet1.0-12.onnx) |INT8|ImageNet|54.3 (0.54)|
|  [MobileNet V2](https://github.com/onnx/models/tree/8e893eb39b131f6d3970be6ebd525327d3df34ea/vision/classification/mobilenet/model/mobilenetv2-12.onnx)   |INT8|ImageNet|71.38 (0.49)|
|  [DenseNet-121](https://github.com/onnx/models/tree/8e893eb39b131f6d3970be6ebd525327d3df34ea/vision/classification/densenet-121/model/densenet-12.onnx)   |INT8|ImageNet|60.16 (0.8)|
|     [VGG-16](https://github.com/onnx/models/tree/8e893eb39b131f6d3970be6ebd525327d3df34ea/vision/classification/vgg/model/vgg16-12.onnx)      |INT8|ImageNet|72.02 (0.0)|

### Object Detection

|   ONNX Model    |Compression algorithm| Dataset |mAP (drop) %|
|:---------------:| :---: | :---: | :---: |
|   [SSD1200](https://github.com/onnx/models/tree/8e893eb39b131f6d3970be6ebd525327d3df34ea/vision/object_detection_segmentation/ssd/model/ssd-12.onnx)   |INT8|COCO2017|20.17 (0.17)|
| [Tiny-YOLOv2](https://github.com/onnx/models/tree/8e893eb39b131f6d3970be6ebd525327d3df34ea/vision/object_detection_segmentation/tiny-yolov2/model/tinyyolov2-8.onnx) |INT8|VOC12|29.03 (0.23)|