# Post-Training Quantization (PTQ) using ONNXRuntime

This examples shows how to quantize ONNX formated NN model (FP32) into the quantized NN model (INT8) using NNCF PTQ API with ONNXRuntime framework.

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
                onnx_ptq_experimental:dev

(container) $ nncf/examples/experimental/onnx/run_ptq_onnx_models.sh [classification|det_and_seg] /onnx-models /output /omz_data $NUMBER_OF_SAMPLES
```

You have to choose the model type between `[classification|det_and_seg]`.
`NUMBER_OF_SAMPLES` is an integer value which is the number of samples required for PTQ parameter calibrations and accuracy checks.
For examples, to run with `NUMBER_OF_SAMPLES=500`, you can command as follows.

1. Classification models

```bash
(container) $ nncf/examples/experimental/onnx/run_ptq_onnx_models.sh classification /onnx-models /output /omz_data 500
```

2. Object detection and segmentation models

```bash
(container) $ nncf/examples/run_ptq_onnx_models.sh det_and_seg /onnx-models /output /omz_data 500
```

After benchmark is done, outputs are located in `/output` which is a mounted directory from the host path `<OUTPUT_DIR>`.

### Results

1. Classification models

| name                  | FP32 latency (ms) | INT8 latency (ms) | Latency diff. (FP32/INT8) | FP32 accuracy (%) | INT8 accuracy (%) | Accuracy diff. (%) |
| --------------------- | ----------------- | ----------------- | ------------------------- | ----------------- | ----------------- | ------------------ |
| bvlcalexnet-12        | 6.65              | 2.91              | 2.29                      | 50.33             | 49.67             | 0.67               |
| caffenet-12           | 6.75              | 2.81              | 2.4                       | 54                | 54                | 0                  |
| densenet-12           | 12.7              | NaN               | NaN                       | 60                | NaN               | NaN                |
| efficientnet-lite4-11 | 7.07              | 3.94              | 1.79                      | 77.67             | 78                | \-0.33             |
| googlenet-12          | 6.52              | 5.48              | 1.19                      | 69                | 68.33             | 0.67               |
| inception-v1-12       | 6.54              | 5.5               | 1.19                      | 67                | 67                | 0                  |
| inception-v2-9        | 8.31              | NaN               | NaN                       | 0                 | NaN               | NaN                |
| mobilenetv2-12        | 3.77              | 3.62              | 1.04                      | 73.33             | 71.33             | 2                  |
| resnet50-v1-12        | 10.99             | 6.02              | 1.82                      | 73.33             | 72                | 1.33               |
| resnet50-v2-7         | 13.23             | 5.32              | 2.49                      | 73.67             | 74                | \-0.33             |
| shufflenet-9          | 5.05              | 6.14              | 0.82                      | 48                | 0                 | 48                 |
| shufflenet-v2-12      | 3.88              | NaN               | NaN                       | 68.33             | NaN               | NaN                |
| squeezenet1.0-12      | 2.63              | 2.56              | 1.03                      | 52.33             | 52.33             | 0                  |
| vgg16-12              | 23.56             | 11.48             | 2.05                      | 71.67             | 70.67             | 1                  |
| zfnet512-12           | 9.6               | 4.16              | 2.31                      | 57                | 57.67             | \-0.67             |

<details>
<summary>Performance benchmark using OpenVINO runtime (not ONNXRuntime-OpenVINOExecutionProvider)</summary>
<table>
    <tr>
        <td>name</td>
        <td>FP32 latency (ms)</td>
        <td>INT8 latency (ms)</td>
        <td>Latency diff. (FP32/INT8)</td>
    </tr>
    <tr>
        <td>bvlcalexnet-12</td>
        <td>26.08</td>
        <td>5.54</td>
        <td>4.71</td>
    </tr>
    <tr>
        <td>caffenet-12</td>
        <td>25.9</td>
        <td>5.13</td>
        <td>5.05</td>
    </tr>
    <tr>
        <td>densenet-12</td>
        <td>30.69</td>
        <td>39.13</td>
        <td>0.78</td>
    </tr>
    <tr>
        <td>efficientnet-lite4-11</td>
        <td>19.81</td>
        <td>6.55</td>
        <td>3.02</td>
    </tr>
    <tr>
        <td>googlenet-12</td>
        <td>15.37</td>
        <td>9.43</td>
        <td>1.63</td>
    </tr>
    <tr>
        <td>inception-v1-12</td>
        <td>14.13</td>
        <td>8.62</td>
        <td>1.64</td>
    </tr>
    <tr>
        <td>inception-v2-9</td>
        <td>18.98</td>
        <td>10.23</td>
        <td>1.86</td>
    </tr>
    <tr>
        <td>mobilenetv2-12</td>
        <td>5.73</td>
        <td>1.9</td>
        <td>3.02</td>
    </tr>
    <tr>
        <td>resnet50-v1-12</td>
        <td>32.45</td>
        <td>14.52</td>
        <td>2.23</td>
    </tr>
    <tr>
        <td>resnet50-v2-7</td>
        <td>41.19</td>
        <td>11.1</td>
        <td>3.71</td>
    </tr>
    <tr>
        <td>shufflenet-9</td>
        <td>4.6</td>
        <td>nan</td>
        <td>nan</td>
    </tr>
    <tr>
        <td>shufflenet-v2-12</td>
        <td>3.45</td>
        <td>nan</td>
        <td>nan</td>
    </tr>
    <tr>
        <td>squeezenet1.0-12</td>
        <td>3.79</td>
        <td>1.88</td>
        <td>2.02</td>
    </tr>
    <tr>
        <td>vgg16-12</td>
        <td>160.57</td>
        <td>39.12</td>
        <td>4.1</td>
    </tr>
    <tr>
        <td>zfnet512-12</td>
        <td>43.75</td>
        <td>14.22</td>
        <td>3.08</td>
    </tr>
</table>
</details>

2. Object detection and segmentation models

| name                 | FP32 latency (ms) | INT8 latency (ms) | Latency diff. (FP32/INT8) | FP32 accuracy (%) | INT8 accuracy (%) | Accuracy diff. (%) |
| -------------------- | ----------------- | ----------------- | ------------------------- | ----------------- | ----------------- | ------------------ |
| FasterRCNN-12        | NaN               | NaN               | NaN                       | 37.71             | NaN               | NaN                |
| MaskRCNN-12-det      | NaN               | NaN               | NaN                       | 36.91             | NaN               | NaN                |
| MaskRCNN-12-inst-seg | NaN               | NaN               | NaN                       | 34.27             | NaN               | NaN                |
| ResNet101-DUC-12     | 904.88            | 435.51            | 2.08                      | 71.2              | 70.36             | 0.84               |
| fcn-resnet50-12      | 250.28            | 400.34            | 0.63                      | 38.31             | 38.14             | 0.17               |
| retinanet-9          | 110.18            | NaN               | NaN                       | 18.45             | NaN               | NaN                |
| ssd-12               | 205.84            | NaN               | NaN                       | 22.91             | NaN               | NaN                |
| tiny-yolov3-11       | NaN               | NaN               | NaN                       | 8.68              | NaN               | NaN                |
| tinyyolov2-8         | 10.48             | 6.06              | 1.73                      | 32.34             | 31.78             | 0.56               |
| yolov2-coco-9        | 24.06             | 14.8              | 1.63                      | 21.7              | 22.17             | \-0.47             |
| yolov3-12            | NaN               | NaN               | NaN                       | 31.08             | NaN               | NaN                |
| yolov4               | 41.08             | NaN               | NaN                       | 14.28             | NaN               | NaN                |

<details>
<summary>Performance benchmark using OpenVINO runtime (not ONNXRuntime-OpenVINOExecutionProvider)</summary>
<table>
    <tr>
        <td>name</td>
        <td>FP32 latency (ms)</td>
        <td>INT8 latency (ms)</td>
        <td>Latency diff. (FP32/INT8)</td>
    </tr>
    <tr>
        <td>FasterRCNN-12</td>
        <td>nan</td>
        <td>nan</td>
        <td>nan</td>
    </tr>
    <tr>
        <td>MaskRCNN-12-det*</td>
        <td>nan</td>
        <td>nan</td>
        <td>nan</td>
    </tr>
    <tr>
        <td>MaskRCNN-12-inst-seg*</td>
        <td>nan</td>
        <td>nan</td>
        <td>nan</td>
    </tr>
    <tr>
        <td>ResNet101-DUC-7</td>
        <td>6495.49</td>
        <td>nan</td>
        <td>nan</td>
    </tr>
    <tr>
        <td>fcn-resnet50-12</td>
        <td>nan</td>
        <td>nan</td>
        <td>nan</td>
    </tr>
    <tr>
        <td>retinanet-9</td>
        <td>852.62</td>
        <td>218.22</td>
        <td>3.91</td>
    </tr>
    <tr>
        <td>ssd-12</td>
        <td>1909.57</td>
        <td>nan</td>
        <td>nan</td>
    </tr>
    <tr>
        <td>ssd_mobilenet_v1_12</td>
        <td>nan</td>
        <td>nan</td>
        <td>nan</td>
    </tr>
    <tr>
        <td>tiny-yolov3-11</td>
        <td>nan</td>
        <td>nan</td>
        <td>nan</td>
    </tr>
    <tr>
        <td>tinyyolov2-8</td>
        <td>32.41</td>
        <td>9.38</td>
        <td>3.46</td>
    </tr>
    <tr>
        <td>yolov2-coco-9</td>
        <td>124.68</td>
        <td>45.74</td>
        <td>2.73</td>
    </tr>
    <tr>
        <td>yolov3-12</td>
        <td>nan</td>
        <td>nan</td>
        <td>nan</td>
    </tr>
    <tr>
        <td>yolov4</td>
        <td>281.87</td>
        <td>460.96</td>
        <td>0.61</td>
    </tr>
</table>
</details>

* `nan` means that NNCF PTQ API failed to generate proper quantized onnx model. We are working on these defects.
* `MaskRCNN-12` can be used two task types detection (`det`) and instance segmentation (`inst-seg`).

# Support ONNXRuntime PTQ for Yolov5 models

## Prerequisite

1. Follow the [installation step](#installation)
2. Clone [Yolov5 repository](https://github.com/ultralytics/yolov5.git) and patch it.

```bash
$ cd examples/experimental/onnx/yolov5
$ git init
$ git remote add origin https://github.com/ultralytics/yolov5.git
$ git fetch origin
$ git checkout 34df5032a7d2e83fe3d16770a03bd129b115d184
$ git apply 0001-Add-NNCF-ONNX-PTQ-example-notebook.patch
```

## Run NNCF ONNXRuntime PTQ

After [prerequisite](#prerequisite) is done, you can find `run_notebook.ipynb` notebook file in `examples/experimental/onnx/yolov5`. If you finish running all notebook cells, you will obtain the following PTQ benchmark results.

```
# Model latency
FP32 latency: 23.7ms, INT8 latency: 19.7ms, FP32/INT8: 1.21x
# Model accuracy
FP32 mAP: 37.1%, INT8 mAP: 36.4%, mAP difference: 0.8%
```
