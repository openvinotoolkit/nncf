## Benchmark for ONNX Model Zoo

## Installation
### Pip installation

Firstly, you would better to prepare a Python virtual environment with Python3.8. Then, please refer to [installation guide for developers](../../../CONTRIBUTING.md#experimental-onnxruntime-openvino) to configure the environment.

### Docker image build

You should make an environment including [ONNXRuntime](https://onnxruntime.ai/docs) with [OpenVINOExecutionProvider](https://onnxruntime.ai/docs/execution-providers/OpenVINO-ExecutionProvider.html). We officially support `onnxruntime==1.11.0` and `openvino==2022.1.0`. You can use use a docker image build script we provided in `./docker/onnx/openvinoep/build.sh` to configure the environment easily.

```bash
# Build
$ ./docker/onnx/openvinoep/build.sh
...
Successfully tagged onnx_ptq_experimental:dev

# Check image
$ docker images | grep onnx_ptq_experimental:dev
```

## Benchmark for ONNX Models: Vision

### Prepare dataset

1. Classification models

Because we use [OpenVINO™ Accuracy Checker](https://github.com/openvinotoolkit/open_model_zoo/tree/master/tools/accuracy_checker) tool, you should prepare ILSVRC2012 validation dataset by following the [dataset preparation guide](https://github.com/openvinotoolkit/open_model_zoo/blob/2022.1.0/data/datasets.md#imagenet). After preparation, your dataset directory will be:

```
DATASET_DIR/
+-- ILSVRC2012_img_val/
|   +-- ILSVRC2012_val_00000001.JPEG
|   +-- ILSVRC2012_val_00000002.JPEG
|   +-- ILSVRC2012_val_00000003.JPEG
|   +-- ...
+-- val.txt
```

2. Object detection and segmentation models

We use [COCO](https://github.com/openvinotoolkit/open_model_zoo/blob/2022.1.0/data/datasets.md#common-objects-in-context-coco), [VOC2012](https://github.com/openvinotoolkit/open_model_zoo/blob/2022.1.0/data/datasets.md#visual-object-classes-challenge-2012-voc2012) and [CityScapes](https://github.com/openvinotoolkit/open_model_zoo/blob/cf9003a95ddb742aabea341aa1573c3fa25ebbe1/data/dataset_definitions.yml#L1300-L1307) datasets. Please follow the link to prepare datasets. After preparation, your dataset directory will be:

```
DATASET_DIR/
+-- annotations/ (COCO annotatios)
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

### Prepare models

1. Classification models

You can download models from [ONNX Model Zoo - Image Classification](https://github.com/onnx/models#image_classification).
In this example, you have to prepare 15 classification models.

1. [bvlcalexnet-12](https://github.com/onnx/models/blob/main/vision/classification/alexnet/model/bvlcalexnet-12.onnx)
2. [caffenet-12](https://github.com/onnx/models/blob/main/vision/classification/caffenet/model/caffenet-12.onnx)
3. [densenet-12](https://github.com/onnx/models/blob/main/vision/classification/densenet-121/model/densenet-12.onnx)
4. [efficientnet-lite4-11](https://github.com/onnx/models/blob/main/vision/classification/efficientnet-lite4/model/efficientnet-lite4-11.onnx)
5. [googlenet-12](https://github.com/onnx/models/blob/main/vision/classification/inception_and_googlenet/googlenet/model/googlenet-12.onnx)
6. [inception-v1-12](https://github.com/onnx/models/blob/main/vision/classification/inception_and_googlenet/inception_v1/model/inception-v1-12.onnx)
7. [mobilenetv2-12](https://github.com/onnx/models/blob/main/vision/classification/mobilenet/model/mobilenetv2-12.onnx)
8. [resnet50-v1-12](https://github.com/onnx/models/blob/main/vision/classification/resnet/model/resnet50-v1-12.onnx)
9. [resnet50-v2-7](https://github.com/onnx/models/blob/main/vision/classification/resnet/model/resnet50-v2-7.onnx)
10. [shufflenet-9](https://github.com/onnx/models/blob/main/vision/classification/shufflenet/model/shufflenet-9.onnx)
11. [shufflenet-v2-12](https://github.com/onnx/models/blob/main/vision/classification/shufflenet/model/shufflenet-v2-12.onnx)
12. [squeezenet1.0-12](https://github.com/onnx/models/blob/main/vision/classification/squeezenet/model/squeezenet1.0-12.onnx)
13. [vgg16-12](https://github.com/onnx/models/blob/main/vision/classification/vgg/model/vgg16-12.onnx)
14. [zfnet512-12](https://github.com/onnx/models/blob/main/vision/classification/zfnet-512/model/zfnet512-12.onnx)

All downloaded models are located in the same directory. `MODELS_DIR` should be the following structure.
```
MODELS_DIR/
+-- bvlcalexnet-12.onnx
+-- caffenet-12.onnx
+-- densenet-12.onnx
+-- ...
(Total 15 onnx files)
```

2. Object detection and segmentation models

You can download models from [ONNX Model Zoo - Object Detection & Image Segmentation](https://github.com/onnx/models#object_detection).
In this example, you have to prepare 15 classification models.

- [FasterRCNN-12](https://github.com/onnx/models/blob/main/vision/object_detection_segmentation/faster-rcnn/model/FasterRCNN-12.onnx)
- [MaskRCNN-12](https://github.com/onnx/models/blob/main/vision/object_detection_segmentation/mask-rcnn/model/MaskRCNN-12.onnx)
- [ResNet101-DUC-7](https://github.com/onnx/models/blob/main/vision/object_detection_segmentation/duc/model/ResNet101-DUC-7.onnx)
- [fcn-resnet50-12](https://github.com/onnx/models/blob/main/vision/object_detection_segmentation/fcn/model/fcn-resnet50-12.onnx)
- [retinanet-9](https://github.com/onnx/models/blob/main/vision/object_detection_segmentation/retinanet/model/retinanet-9.onnx)
- [ssd-12](https://github.com/onnx/models/blob/main/vision/object_detection_segmentation/ssd/model/ssd-12.onnx)
- [ssd_mobilenet_v1_12](https://github.com/onnx/models/blob/main/vision/object_detection_segmentation/ssd-mobilenetv1/model/ssd_mobilenet_v1_12.onnx)
- [tiny-yolov3-11](https://github.com/onnx/models/blob/main/vision/object_detection_segmentation/tiny-yolov3/model/tiny-yolov3-11.onnx)
- [tinyyolov2-8](https://github.com/onnx/models/blob/main/vision/object_detection_segmentation/tiny-yolov2/model/tinyyolov2-8.onnx)
- [yolov2-coco-9](https://github.com/onnx/models/blob/main/vision/object_detection_segmentation/yolov2-coco/model/yolov2-coco-9.onnx)
- [yolov3-12](https://github.com/onnx/models/blob/main/vision/object_detection_segmentation/yolov3/model/yolov3-12.onnx)
- [yolov4](https://github.com/onnx/models/blob/main/vision/object_detection_segmentation/yolov4/model/yolov4.onnx)

All downloaded models are located in the same directory. `MODELS_DIR` should be the following structure.
```
MODELS_DIR/
+-- FasterRCNN-12.onnx
+-- MaskRCNN-12.onnx
+-- ResNet101-DUC-7.onnx
+-- ...
(Total 12 onnx files)
```

### Prepare docker image

Please refer to [Docker image build](#docker-image-build) section.

### Run benchmark

```bash
(host)      $ docker run                    \
                -it --rm --name onnx_ptq    \
                -v <DATASET_DIR>:/omz_data  \
                -v <MODEL_DIR>:/onnx-models \
                -v <OUTPUT_DIR>:/output     \
                onnx_ptq_experimental:dev

(container) $ nncf/tests/onnx/benchmarking/run_ptq_onnx_models.sh [classification|object_detection_segmentation] /onnx-models /output /omz_data $NUMBER_OF_SAMPLES
```

You have to choose the model type between `[classification|det_and_seg]`.
`NUMBER_OF_SAMPLES` is an integer value which is the number of samples required for PTQ parameter calibrations and accuracy checks.
For examples, to run with `NUMBER_OF_SAMPLES=500`, you can command as follows.

1. Classification models

```bash
(container) $ nncf/tests/onnx/benchmarking/run_ptq_onnx_models.sh classification /onnx-models /output /omz_data 500
```

2. Object detection and segmentation models

```bash
(container) $ nncf/tests/onnx/benchmarking/run_ptq_onnx_models.sh object_detection_segmentation /onnx-models /output /omz_data 500
```

After benchmark is done, outputs are located in `/output` which is a mounted directory from the host path `<OUTPUT_DIR>`.

### Results

1. Classification models

| name                              | FP32 accuracy (%) | INT8 accuracy (%) | Accuracy diff. (%) |
| --------------------------------- | ----------------- | ----------------- | ------------------ |
| bvlcalexnet-12                    | 50.33             | 49.67             | 0.67               |
| caffenet-12                       | 54                | 54                | 0                  |
| densenet-12                       | 60                | 58.33             | 1.67               |
| efficientnet-lite4-11             | 77.67             | 78                | \-0.33             |
| googlenet-12                      | 69                | 68.33             | 0.67               |
| inception-v1-12                   | 67                | 67                | 0                  |
| mobilenetv2-12                    | 73.33             | 71.33             | 2                  |
| resnet50-v1-12                    | 73.33             | 72                | 1.33               |
| resnet50-v2-7                     | 73.67             | 74                | \-0.33             |
| shufflenet-9 :cloud:              | 48                | 46.33             | 1.67               |
| shufflenet-v2-12 :cloud:          | 68.33             | 67.33             | 1                  |
| squeezenet1.0-12                  | 52.33             | 52.33             | 0                  |
| vgg16-12                          | 71.67             | 70.67             | 1                  |
| zfnet512-12                       | 57                | 57.67             | \-0.67             |

2. Object detection and segmentation models

| name                          | FP32 accuracy (%) | INT8 accuracy (%) | Accuracy diff. (%) |
| ----------------------------- | ----------------- | ----------------- | ------------------ |
| FasterRCNN-12                 | 37.71             | 37.45             | 0.26               |
| MaskRCNN-12-det               | 36.91             | 37.06             | \-0.15             |
| MaskRCNN-12-inst-seg          | 34.27             | 34.28             | \-0.01             |
| ResNet101-DUC-12              | 71.2              | 70.36             | 0.84               |
| fcn-resnet50-12               | 38.31             | 38.14             | 0.17               |
| retinanet-9                   | 18.45             | 18.39             | 0.06               |
| ssd-12                        | 22.91             | 22.54             | 0.37               |
| ssd_mobilenet_v1_12 :cloud:   | -                 | -                 | -                  |
| tiny-yolov3-11 :cloud:        | 8.68              | 7.97              | 0.71               |
| tinyyolov2-8                  | 32.34             | 31.78             | 0.56               |
| yolov2-coco-9                 | 21.7              | 22.17             | \-0.47             |
| yolov3-12 :cloud:             | 31.08             | 29.01             | 2.07               |
| yolov4 :cloud:                | 14.28             | 13.97             | 0.31               |

* `nan` means that NNCF PTQ API failed to generate proper quantized onnx model. We are working on these defects.
* `MaskRCNN-12` can be used two task types detection (`det`) and instance segmentation (`inst-seg`).
* Failure types:
    1. :umbrella: - A quantized model shows too much model accuracy degradation.
    2. :cloud: - A quantized model cannot be executed on `OpenVINOExecutionProvider(onnxruntime-openvino==1.11.0)`, but it can be executed on `CPUExecutionProvider` (`shufflenet-9` can be executed on `OpenVINOExecutionProvider`, but drops too much accuracy).
