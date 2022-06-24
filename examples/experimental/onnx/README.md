# Post-Training Quantization (PTQ) using ONNXRuntime

This examples shows how to quantize ONNX formated NN model (FP32) into the quantized NN model (INT8) using NNCF PTQ API with ONNXRuntime framework.

## Docker image build

You should make an environment including [ONNXRuntime](https://onnxruntime.ai/docs) with [OpenVINOExecutionProvider](https://onnxruntime.ai/docs/execution-providers/OpenVINO-ExecutionProvider.html). We officially support `onnxruntime==1.11.0` and `openvino==2022.1.0`. You can use use a docker image build script we provided in `./docker/onnx/openvinoep/build.sh` to configure the environment easily.

```bash
# Build
$ ./docker/onnx/openvinoep/build.sh
...
Successfully tagged onnx_ptq_experimental:latest

# Check image
$ docker images | grep onnx_ptq_experimental:latest
```

## Run NNCF PTQ for ONNXRuntime for your model

Please refer to guides:

1. [Classification models](classification/README.md)
2. [Semantic segmenantation models](semantic_segmentation/README.md)

## Benchmark for ONNX Model Zoo

[ONNX Model Zoo](https://github.com/onnx/models) provides is an open standard format for popular deep NN models with their pretrained weights. In this examples, we will quantize ONNX Model ZOO models. After quantization, model accuracy and model latency are compared between the original model (FP32) and quantized model (INT8).

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
7. [inception-v2-9](https://github.com/onnx/models/blob/main/vision/classification/inception_and_googlenet/inception_v2/model/inception-v2-9.onnx)
8. [mobilenetv2-12](https://github.com/onnx/models/blob/main/vision/classification/mobilenet/model/mobilenetv2-12.onnx)
9. [resnet50-v1-12](https://github.com/onnx/models/blob/main/vision/classification/resnet/model/resnet50-v1-12.onnx)
10. [resnet50-v2-7](https://github.com/onnx/models/blob/main/vision/classification/resnet/model/resnet50-v2-7.onnx)
11. [shufflenet-9](https://github.com/onnx/models/blob/main/vision/classification/shufflenet/model/shufflenet-9.onnx)
12. [shufflenet-v2-12](https://github.com/onnx/models/blob/main/vision/classification/shufflenet/model/shufflenet-v2-12.onnx)
13. [squeezenet1.0-12](https://github.com/onnx/models/blob/main/vision/classification/squeezenet/model/squeezenet1.0-12.onnx)
14. [vgg16-12](https://github.com/onnx/models/blob/main/vision/classification/vgg/model/vgg16-12.onnx)
15. [zfnet512-12](https://github.com/onnx/models/blob/main/vision/classification/zfnet-512/model/zfnet512-12.onnx)

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

You can download models from [ONNX Model Zoo - Image Classification](https://github.com/onnx/models#image_classification).
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
                onnx_ptq_experimental:latest
(container) $ ./examples/run_ptq_onnx_models.sh [classification|det_and_seg] /onnx-models /output /omz_data $NUMBER_OF_SAMPLES
```

You have to choose the model type between `[classification|det_and_seg]`.
`NUMBER_OF_SAMPLES` is an integer value which is the number of samples required for PTQ parameter calibrations and accuracy checks.
For examples, to run with `NUMBER_OF_SAMPLES=500`, you can command as follows.

1. Classification models

```bash
(container) $ ./examples/run_ptq_onnx_models.sh classification /onnx-models /output /omz_data 500
```

2. Object detection and segmentation models

```bash
(container) $ ./examples/run_ptq_onnx_models.sh det_and_seg /onnx-models /output /omz_data 500
```

After benchmark is done, outputs are located in `/output` which is a mounted directory from the host path `<OUTPUT_DIR>`.

### Results

1. Classification models

| name                  |   FP32 latency (ms) |   INT8 latency (ms) |   Latency diff. (FP32/INT8) |   FP32 accuracy (%) |   INT8 accuracy (%) |  Accuracy diff. (%) |
|:----------------------|-------------------:|--------------------:|---------------:|--------------------:|---------------------:|----------------:|
| bvlcalexnet-12        |              26.08 |                5.54 |           4.71 |               50.33 |                49.67 |            0.67 |
| caffenet-12           |              25.90 |                5.13 |           5.05 |               54.00 |                54.00 |            0.00 |
| densenet-12           |              30.69 |               39.13 |           0.78 |               60.00 |               nan    |          nan    |
| efficientnet-lite4-11 |              19.81 |                6.55 |           3.02 |               77.33 |                78.33 |           -1.00 |
| googlenet-12          |              15.37 |                9.43 |           1.63 |               69.00 |                67.67 |            1.33 |
| inception-v1-12       |              14.13 |                8.62 |           1.64 |               67.00 |                66.67 |            0.33 |
| inception-v2-9        |              18.98 |               10.23 |           1.86 |                0.00 |               nan    |          nan    |
| mobilenetv2-12        |               5.73 |                1.90 |           3.02 |               73.33 |                72.33 |            1.00 |
| resnet50-v1-12        |              32.45 |               14.52 |           2.23 |               73.33 |                72.33 |            1.00 |
| resnet50-v2-7         |              41.19 |               11.10 |           3.71 |               73.67 |                73.33 |            0.33 |
| shufflenet-9          |               4.60 |              nan    |         nan    |               48.00 |                 0.00 |           48.00 |
| shufflenet-v2-12      |               3.45 |              nan    |         nan    |               68.33 |               nan    |          nan    |
| squeezenet1.0-12      |               3.79 |                1.88 |           2.02 |               52.33 |                52.67 |           -0.33 |
| vgg16-12              |             160.57 |               39.12 |           4.10 |               71.33 |                70.67 |            0.67 |
| zfnet512-12           |              43.75 |               14.22 |           3.08 |               57.00 |                57.67 |           -0.67 |

2. Object detection and segmentation models

| name                 |  FP32 latency (ms) |   INT8 latency (ms) |   Latency diff. (FP32/INT8) |   FP32 accuracy (%) |   INT8 accuracy (%) |   Accuracy diff. (%) |
|:---------------------|-------------------:|--------------------:|---------------:|--------------------:|---------------------:|----------------:|
| FasterRCNN-12        |             nan    |              nan    |         nan    |               37.71 |               nan    |          nan    |
| MaskRCNN-12-det*     |             nan    |              nan    |         nan    |               36.91 |               nan    |          nan    |
| MaskRCNN-12-inst-seg*|             nan    |              nan    |         nan    |               34.27 |               nan    |          nan    |
| ResNet101-DUC-7      |            6495.49 |              nan    |         nan    |               71.21 |               nan    |          nan    |
| fcn-resnet50-12      |             nan    |              nan    |         nan    |               38.31 |                38.14 |            0.17 |
| retinanet-9          |             852.62 |              218.22 |           3.91 |               18.45 |               nan    |          nan    |
| ssd-12               |            1909.57 |              nan    |         nan    |               22.91 |               nan    |          nan    |
| ssd_mobilenet_v1_12  |             nan    |              nan    |         nan    |              nan    |               nan    |          nan    |
| tiny-yolov3-11       |             nan    |              nan    |         nan    |                8.68 |               nan    |          nan    |
| tinyyolov2-8         |              32.41 |                9.38 |           3.46 |               32.34 |                31.97 |            0.36 |
| yolov2-coco-9        |             124.68 |               45.74 |           2.73 |               21.70 |                22.17 |           -0.47 |
| yolov3-12            |             nan    |              nan    |         nan    |               31.08 |               nan    |          nan    |
| yolov4               |             281.87 |              460.96 |           0.61 |               14.28 |               nan    |          nan    |

* `nan` means that NNCF PTQ API failed to generate proper quantized onnx model. We are working on these defects.
* `MaskRCNN-12` can be used two task types detection (`det`) and instance segmentation (`inst-seg`).
