# Benchmark for ONNX Model Zoo

## Installation

### Pip installation

At this point it is assumed that you have already installed NNCF. You can find information on installation
NNCF [here](https://github.com/openvinotoolkit/nncf#user-content-installation).

To work with the example you should install the corresponding Python package dependencies:

```bash
pip install -r requirements.txt
```

## Benchmark for ONNX Models: Vision

In a benchmarking section NNCF
uses [OpenVINO™ Accuracy Checker](https://github.com/openvinotoolkit/open_model_zoo/tree/master/tools/accuracy_checker)
tool to preprocess data for quantization parameters calibration and for final accuracy validation.
The benchmarking supports the following models:

- Classification

1. [bvlcalexnet-12](https://github.com/onnx/models/blob/5faef4c33eba0395177850e1e31c4a6a9e634c82/vision/classification/alexnet/model/bvlcalexnet-12.onnx)
2. [caffenet-12](https://github.com/onnx/models/blob/5faef4c33eba0395177850e1e31c4a6a9e634c82/vision/classification/caffenet/model/caffenet-12.onnx)
3. [densenet-12](https://github.com/onnx/models/blob/5faef4c33eba0395177850e1e31c4a6a9e634c82/vision/classification/densenet-121/model/densenet-12.onnx)
4. [efficientnet-lite4-11](https://github.com/onnx/models/blob/5faef4c33eba0395177850e1e31c4a6a9e634c82/vision/classification/efficientnet-lite4/model/efficientnet-lite4-11.onnx)
5. [googlenet-12](https://github.com/onnx/models/blob/5faef4c33eba0395177850e1e31c4a6a9e634c82/vision/classification/inception_and_googlenet/googlenet/model/googlenet-12.onnx)
6. [inception-v1-12](https://github.com/onnx/models/blob/5faef4c33eba0395177850e1e31c4a6a9e634c82/vision/classification/inception_and_googlenet/inception_v1/model/inception-v1-12.onnx)
7. [mobilenetv2-12](https://github.com/onnx/models/blob/5faef4c33eba0395177850e1e31c4a6a9e634c82/vision/classification/mobilenet/model/mobilenetv2-12.onnx)
8. [resnet50-v1-12](https://github.com/onnx/models/blob/5faef4c33eba0395177850e1e31c4a6a9e634c82/vision/classification/resnet/model/resnet50-v1-12.onnx)
9. [resnet50-v2-7](https://github.com/onnx/models/blob/5faef4c33eba0395177850e1e31c4a6a9e634c82/vision/classification/resnet/model/resnet50-v2-7.onnx)
10. [shufflenet-9](https://github.com/onnx/models/blob/5faef4c33eba0395177850e1e31c4a6a9e634c82/vision/classification/shufflenet/model/shufflenet-9.onnx)
11. [shufflenet-v2-12](https://github.com/onnx/models/blob/5faef4c33eba0395177850e1e31c4a6a9e634c82/vision/classification/shufflenet/model/shufflenet-v2-12.onnx)
12. [squeezenet1.0-12](https://github.com/onnx/models/blob/5faef4c33eba0395177850e1e31c4a6a9e634c82/vision/classification/squeezenet/model/squeezenet1.0-12.onnx)
13. [vgg16-12](https://github.com/onnx/models/blob/5faef4c33eba0395177850e1e31c4a6a9e634c82/vision/classification/vgg/model/vgg16-12.onnx)
14. [zfnet512-12](https://github.com/onnx/models/blob/5faef4c33eba0395177850e1e31c4a6a9e634c82/vision/classification/zfnet-512/model/zfnet512-12.onnx)

- Object detection and segmentation models

1. [FasterRCNN-12](https://github.com/onnx/models/blob/5faef4c33eba0395177850e1e31c4a6a9e634c82/vision/object_detection_segmentation/faster-rcnn/model/FasterRCNN-12.onnx)
2. [MaskRCNN-12](https://github.com/onnx/models/blob/5faef4c33eba0395177850e1e31c4a6a9e634c82/vision/object_detection_segmentation/mask-rcnn/model/MaskRCNN-12.onnx)
3. [ResNet101-DUC-7](https://github.com/onnx/models/blob/5faef4c33eba0395177850e1e31c4a6a9e634c82/vision/object_detection_segmentation/duc/model/ResNet101-DUC-7.onnx)
4. [fcn-resnet50-12](https://github.com/onnx/models/blob/5faef4c33eba0395177850e1e31c4a6a9e634c82/vision/object_detection_segmentation/fcn/model/fcn-resnet50-12.onnx)
5. [retinanet-9](https://github.com/onnx/models/blob/5faef4c33eba0395177850e1e31c4a6a9e634c82/vision/object_detection_segmentation/retinanet/model/retinanet-9.onnx)
6. [ssd-12](https://github.com/onnx/models/blob/5faef4c33eba0395177850e1e31c4a6a9e634c82/vision/object_detection_segmentation/ssd/model/ssd-12.onnx)
7. [ssd_mobilenet_v1_12](https://github.com/onnx/models/blob/5faef4c33eba0395177850e1e31c4a6a9e634c82/vision/object_detection_segmentation/ssd-mobilenetv1/model/ssd_mobilenet_v1_12.onnx)
8. [tiny-yolov3-11](https://github.com/onnx/models/blob/5faef4c33eba0395177850e1e31c4a6a9e634c82/vision/object_detection_segmentation/tiny-yolov3/model/tiny-yolov3-11.onnx)
9. [tinyyolov2-8](https://github.com/onnx/models/blob/5faef4c33eba0395177850e1e31c4a6a9e634c82/vision/object_detection_segmentation/tiny-yolov2/model/tinyyolov2-8.onnx)
10. [yolov2-coco-9](https://github.com/onnx/models/blob/5faef4c33eba0395177850e1e31c4a6a9e634c82/vision/object_detection_segmentation/yolov2-coco/model/yolov2-coco-9.onnx)
11. [yolov3-12](https://github.com/onnx/models/blob/5faef4c33eba0395177850e1e31c4a6a9e634c82/vision/object_detection_segmentation/yolov3/model/yolov3-12.onnx)
12. [yolov4](https://github.com/onnx/models/blob/5faef4c33eba0395177850e1e31c4a6a9e634c82/vision/object_detection_segmentation/yolov4/model/yolov4.onnx)

You can find the Accuracy Checker configs that are used for particular models
in [classification](./classification/onnx_models_configs)
and [object_detection_segmentation](./object_detection_segmentation/onnx_models_configs)

## Steps to run benchmarking

### 1. Prepare dataset

- Classification models

Because we
use [OpenVINO™ Accuracy Checker](https://github.com/openvinotoolkit/open_model_zoo/tree/master/tools/accuracy_checker)
tool, you should prepare ILSVRC2012 validation dataset by following
the [dataset preparation guide](https://github.com/openvinotoolkit/open_model_zoo/blob/2022.1.0/data/datasets.md#imagenet)
. After preparation, your dataset directory will be:

```text
DATASET_DIR/
+-- ILSVRC2012_img_val/
|   +-- ILSVRC2012_val_00000001.JPEG
|   +-- ILSVRC2012_val_00000002.JPEG
|   +-- ILSVRC2012_val_00000003.JPEG
|   +-- ...
+-- val.txt
```

- Object detection and segmentation models

We
use [COCO](https://github.com/openvinotoolkit/open_model_zoo/blob/2022.1.0/data/datasets.md#common-objects-in-context-coco)
, [VOC2012](https://github.com/openvinotoolkit/open_model_zoo/blob/2022.1.0/data/datasets.md#visual-object-classes-challenge-2012-voc2012)
and [CityScapes](https://github.com/openvinotoolkit/open_model_zoo/blob/cf9003a95ddb742aabea341aa1573c3fa25ebbe1/data/dataset_definitions.yml#L1300-L1307)
datasets. Please follow the link to prepare datasets. After preparation, your dataset directory will be:

```text
DATASET_DIR/
+-- annotations/ (COCO annotations)
|   +-- instances_val2017.json
|   +-- ...
+-- val2017/ (COCO images)
|   +-- 000000000139.jpg
|   +-- ...
+-- VOCdevkit/
|   +-- VOC2012/ (VOC2012 datasets)
|   |   +-- Annotations/
|   |   +-- JPEGImages/
|   |   +-- ...
+-- Cityscapes/
|   +-- data/
|   |   +-- gtFine/
|   |   +-- imgsFine/
|   |   +-- ...
```

### 2. Run benchmark

You can run the benchmarking for particular model with the following command:

`python run_ptq.py -c <config path> -m <model path> -o <output dir> -d <dataset_definitions.yml path> -s <dataset path>`

### Results

#### 1. Classification models

| Model Name              | Dataset  | FP32 Accuracy (%) | INT8 accuracy (%) | Accuracy Drop (%) |
|-------------------------|----------|-------------------|-------------------|-------------------|
| bvlcalexnet-12          | ImageNet | 52.02             | 51.96             | 0.06              |
| caffenet-12             | ImageNet | 54.26             | 54.22             | 0.04              |
| densenet-12             | ImageNet | 60.96             | 60.16             | 0.8               |
| efficientnet-lite4-11   | ImageNet | 77.58             | 77.43             | 0.15              |
| googlenet-12            | ImageNet | 66.67             | 66.36             | 0.31              |
| inception-v1-12         | ImageNet | 65.21             | 64.87             | 0.34              |
| mobilenetv2-12          | ImageNet | 71.87             | 71.38             | 0.49              |
| resnet50-v1-12          | ImageNet | 74.11             | 73.92             | 0.19              |
| resnet50-v2-7           | ImageNet | 74.84             | 74.63             | 0.21              |
| shufflenet-9            | ImageNet | 47.43             | 47.25             | 0.18              |
| shufflenet-v2-12        | ImageNet | 69.36             | 68.93             | 0.43              |
| squeezenet1.0-12        | ImageNet | 54.84             | 54.3              | 0.54              |
| vgg16-12                | ImageNet | 72.02             | 72.02             | 0.0               |
| zfnet512-12             | ImageNet | 58.57             | 58.53             | 0.04              |

#### 2. Object detection and segmentation models

| Model Name           | Dataset   | FP32 mAP (%) | INT8 mAP (%)  | mAP diff. (%) |
|----------------------|-----------|--------------|---------------|---------------|
| FasterRCNN-12        | COCO2017  | 34.93        | 34.67         | 0.26          |
| MaskRCNN-12-det      | COCO2017  | -            | -             | -             |
| MaskRCNN-12-inst-seg | COCO2017  | -            | -             | -             |
| ResNet101-DUC-12     | Citycapes | 61.91        | 61.09         | 0.82          |
| fcn-resnet50-12      | COCO2017  | 38.31        | 38.14         | 0.17          |
| retinanet-9          | COCO2017  | 15.03        | 15.14         | -0.11         |
| ssd-12               | COCO2017  | 20.34        | 20.17         | 0.17          |
| ssd_mobilenet_v1_12  | COCO2017  | -            | -             | -             |
| tiny-yolov3-11       | COCO2017  | 7.83         | 7.51          | 0.32          |
| tinyyolov2-8         | VOC2012   | 29.27        | 29.03         | 0.23          |
| yolov2-coco-9        | COCO2017  | 18.24        | 18.8          | -0.57         |
| yolov3-12            | COCO2017  | 29.17        | 27.12         | 2.04          |
| yolov4               | COCO2017  | 10.0         | 9.71          | 0.31          |
