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

## Benchmark for ONNX Models: Vision - Classification

### Prepare dataset

Because we use [OpenVINO™ Accuracy Checker](https://github.com/openvinotoolkit/open_model_zoo/tree/master/tools/accuracy_checker) tool, you should prepare ILSVRC2012 validation dataset by following the [dataset preparation guide](https://github.com/openvinotoolkit/open_model_zoo/blob/2022.1.0/data/datasets.md#imagenet). After preparation, your dataset directory will be:

```
DATASET_DIR
+-- ILSVRC2012_img_val
|   +-- ILSVRC2012_val_00000001.JPEG
|   +-- ILSVRC2012_val_00000002.JPEG
|   +-- ILSVRC2012_val_00000003.JPEG
|   +-- ...
+-- val.txt
```

### Prepare models

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
MODELS_DIR
+-- bvlcalexnet-12.onnx
+-- caffenet-12.onnx
+-- densenet-12.onnx
+-- ... 
(Total 15 onnx files)
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
(container) $ ./examples/run_ptq_onnx_models.sh /onnx-models /output $NUMBER_OF_SAMPLES
```

`NUMBER_OF_SAMPLES` is an integer value which is the number of samples required for PTQ parameter calibrations and accuracy checks. For examples, to run with `NUMBER_OF_SAMPLES=500`, you can command as follows.

```bash
(container) $ ./examples/run_ptq_onnx_models.sh /onnx-models /output 500
```

After benchmark is done, outputs are located in `/output` which is a mounted directory from the host path `<OUTPUT_DIR>`.

### Results

| name                  |   FP32 latency (ms) |   INT8 latency (ms) |   Latency diff. (FP32/INT8) |   FP32 accuracy (%) |   INT8 accuracy (%) |  Accuracy diff. (%) |
|:----------------------|-------------------:|--------------------:|---------------:|--------------------:|---------------------:|----------------:|
| bvlcalexnet-12        |              26.12 |                5.56 |           4.70 |               50.20 |                49.80 |            0.40 |
| caffenet-12           |              26.11 |                5.17 |           5.05 |               54.40 |                54.40 |            0.00 |
| densenet-12           |              30.39 |               38.85 |           0.78 |               59.00 |               nan    |          nan    |
| efficientnet-lite4-11 |              18.44 |                6.45 |           2.86 |               77.80 |                77.60 |            0.20 |
| googlenet-12          |              15.31 |                9.26 |           1.65 |               68.40 |                67.60 |            0.80 |
| inception-v1-12       |              14.02 |              nan    |         nan    |               67.20 |               nan    |          nan    |
| inception-v2-9        |              18.75 |               10.02 |           1.87 |                0.00 |               nan    |          nan    |
| mobilenetv2-12        |               5.58 |                1.86 |           3.00 |               71.80 |                71.20 |            0.60 |
| resnet50-v1-12        |              32.11 |               14.30 |           2.25 |               73.60 |                72.80 |            0.80 |
| resnet50-v2-7         |              41.09 |               11.05 |           3.72 |               73.80 |                74.00 |           -0.20 |
| shufflenet-9          |               4.58 |              nan    |         nan    |               46.60 |                 0.00 |           46.60 |
| shufflenet-v2-12      |               3.40 |              nan    |         nan    |               68.60 |               nan    |          nan    |
| squeezenet1.0-12      |               3.76 |                1.85 |           2.03 |               53.20 |                53.60 |           -0.40 |
| vgg16-12              |             159.44 |               38.75 |           4.11 |               70.80 |                70.60 |            0.20 |
| zfnet512-12           |              43.68 |               14.46 |           3.02 |               58.60 |                59.00 |           -0.40 |

* `nan` means that NNCF PTQ API failed to generate proper quantized onnx model. We are working on these defects.
