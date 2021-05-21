# Image Classification Sample

This sample demonstrates a DL model compression in case of an image-classification problem. The sample consists of basic steps such as DL model initialization, dataset preparation, training loop over epochs, training and validation steps. The sample receives a configuration file where the training schedule, hyper-parameters, and compression settings are defined.

## Features

- Torchvision models (ResNets, VGG, Inception, etc.) and datasets (ImageNet, CIFAR 10, CIFAR 100) support
- Custom models support
- Configuration file examples for sparsity, quantization, and quantization with sparsity
- Export to ONNX that is supported by the OpenVINO™ toolkit
- DataParallel and DistributedDataParallel modes
- Tensorboard-compatible output

## Installation

To work with the sample you should install the corresponding Python package dependencies

```
pip install -r examples/torch/requirements.txt
```

## Quantize FP32 Pretrained Model

This scenario demonstrates quantization with fine-tuning of MobileNet v2 on the ImageNet dataset.

#### Dataset Preparation

To prepare the ImageNet dataset, refer to the following [tutorial](https://github.com/pytorch/examples/tree/master/imagenet).

#### Run Classification Sample

- If you did not install the package, add the repository root folder to the `PYTHONPATH` environment variable.
- Go to the `examples/torch/classification` folder.
- Run the following command to start compression with fine-tuning on GPUs:
    ```
    python main.py -m train --config configs/quantization/mobilenet_v2_imagenet_int8.json --data /data/imagenet/ --log-dir=../../results/quantization/mobilenet_v2_int8/
    ```
    It may take a few epochs to get the baseline accuracy results.
- Use the `--multiprocessing-distributed` flag to run in the distributed mode.
- Use the `--resume` flag with the path to a previously saved model to resume training.
- For Torchvision-supported image classification models, set `"pretrained": true` inside the NNCF config JSON file supplied via `--config` to initialize the model to be compressed with Torchvision-supplied pretrained weights, or, alternatively:
- Use the `--weights` flag with the path to a compatible PyTorch checkpoint in order to load all matching weights from the checkpoint into the model - useful if you need to start compression-aware training from a previously trained uncompressed (FP32) checkpoint instead of performing compression-aware training from scratch.

#### Validate Your Model Checkpoint

To estimate the test scores of your model checkpoint, use the following command:
```
python main.py -m test --config=configs/quantization/mobilenet_v2_imagenet_int8.json --resume <path_to_trained_model_checkpoint>
```
To validate an FP32 model checkpoint, make sure the compression algorithm settings are empty in the configuration file or `pretrained=True` is set.

**WARNING**: The samples use `torch.load` functionality for checkpoint loading which, in turn, uses pickle facilities by default which are known to be vulnerable to arbitrary code execution attacks. **Only load the data you trust**

#### Export Compressed Model

To export trained model to the ONNX format, use the following command:
```
python main.py -m test --config=configs/quantization/mobilenet_v2_imagenet_int8.json --resume=../../results/quantization/mobilenet_v2_int8/6/checkpoints/epoch_1.pth --to-onnx=../../results/mobilenet_v2_int8.onnx
```

#### Export to OpenVINO™ Intermediate Representation (IR)

To export a model to the OpenVINO IR and run it using the Intel® Deep Learning Deployment Toolkit, refer to this [tutorial](https://software.intel.com/en-us/openvino-toolkit).

### Results for quantization

|Model|Compression algorithm|Dataset|PyTorch compressed accuracy|NNCF config file|PyTorch checkpoint|
| :---: | :---: | :---: | :---: | :---: | :---: |
|ResNet-50|None|ImageNet|76.16|[resnet50_imagenet.json](configs/quantization/resnet50_imagenet.json)|-|
|ResNet-50|INT8|ImageNet|76.42|[resnet50_imagenet_int8.json](configs/quantization/resnet50_imagenet_int8.json)|[Link](https://storage.openvinotoolkit.org/repositories/nncf/models/develop/resnet50_imagenet_int8.pth)|
|ResNet-50|INT8 (per-tensor only)|ImageNet|76.37|[resnet50_imagenet_int8_per_tensor.json](configs/quantization/resnet50_imagenet_int8_per_tensor.json)|[Link](https://storage.openvinotoolkit.org/repositories/nncf/models/develop/resnet50_imagenet_int8_per_tensor.pth)|
|ResNet-50|Mixed, 44.8% INT8 / 55.2% INT4|ImageNet|76.2|[resnet50_imagenet_mixed_int_manual.json](configs/mixed_precision/resnet50_imagenet_mixed_int_manual.json)|[Link](https://storage.openvinotoolkit.org/repositories/nncf/models/develop/resnet50_imagenet_int4_int8.pth)|
|ResNet-50|INT8 + Sparsity 61% (RB)|ImageNet|75.43|[resnet50_imagenet_rb_sparsity_int8.json](configs/sparsity_quantization/resnet50_imagenet_rb_sparsity_int8.json)|[Link](https://storage.openvinotoolkit.org/repositories/nncf/models/develop/resnet50_imagenet_rb_sparsity_int8.pth)|
|ResNet-50|INT8 + Sparsity 50% (RB)|ImageNet|75.55|[resnet50_imagenet_rb_sparsity50_int8.json](configs/sparsity_quantization/resnet50_imagenet_rb_sparsity50_int8.json)|[Link](https://storage.openvinotoolkit.org/repositories/nncf/models/develop/resnet50_imagenet_rb_sparsity50_int8.pth)|
|Inception V3|None|ImageNet|77.34|[inception_v3_imagenet.json](configs/quantization/inception_v3_imagenet.json)|-|
|Inception V3|INT8|ImageNet|78.25|[inception_v3_imagenet_int8.json](configs/quantization/inception_v3_imagenet_int8.json)|[Link](https://storage.openvinotoolkit.org/repositories/nncf/models/develop/inception_v3_imagenet_int8.pth)|
|Inception V3|INT8 + Sparsity 61% (RB)|ImageNet|77.58|[inception_v3_imagenet_rb_sparsity_int8.json](configs/sparsity_quantization/inception_v3_imagenet_rb_sparsity_int8.json)|[Link](https://storage.openvinotoolkit.org/repositories/nncf/models/develop/inception_v3_imagenet_rb_sparsity_int8.pth)|
|MobileNet V2|None|ImageNet|71.93|[mobilenet_v2_imagenet.json](configs/quantization/mobilenet_v2_imagenet.json)|-|
|MobileNet V2|INT8|ImageNet|71.35|[mobilenet_v2_imagenet_int8.json](configs/quantization/mobilenet_v2_imagenet_int8.json)|[Link](https://storage.openvinotoolkit.org/repositories/nncf/models/develop/mobilenet_v2_imagenet_int8.pth)|
|MobileNet V2|INT8 (per-tensor only)|ImageNet|71.3|[mobilenet_v2_imagenet_int8_per_tensor.json](configs/quantization/mobilenet_v2_imagenet_int8_per_tensor.json)|[Link](https://storage.openvinotoolkit.org/repositories/nncf/models/develop/mobilenet_v2_imagenet_int8_per_tensor.pth)|
|MobileNet V2|Mixed, 46.6% INT8 / 53.4% INT4|ImageNet|70.92|[mobilenet_v2_imagenet_mixed_int_manual.json](configs/mixed_precision/mobilenet_v2_imagenet_mixed_int_manual.json)|[Link](https://storage.openvinotoolkit.org/repositories/nncf/models/develop/mobilenet_v2_imagenet_int4_int8.pth)|
|MobileNet V2|INT8 + Sparsity 52% (RB)|ImageNet|71.11|[mobilenet_v2_imagenet_rb_sparsity_int8.json](configs/sparsity_quantization/mobilenet_v2_imagenet_rb_sparsity_int8.json)|[Link](https://storage.openvinotoolkit.org/repositories/nncf/models/develop/mobilenet_v2_imagenet_rb_sparsity_int8.pth)|
|SqueezeNet V1.1|None|ImageNet|58.24|[squeezenet1_1_imagenet.json](configs/quantization/squeezenet1_1_imagenet.json)|-|
|SqueezeNet V1.1|INT8|ImageNet|58.28|[squeezenet1_1_imagenet_int8.json](configs/quantization/squeezenet1_1_imagenet_int8.json)|[Link](https://storage.openvinotoolkit.org/repositories/nncf/models/develop/squeezenet1_1_imagenet_int8.pth)|
|SqueezeNet V1.1|INT8 (per-tensor only)|ImageNet|58.26|[squeezenet1_1_imagenet_int8_per_tensor.json](configs/quantization/squeezenet1_1_imagenet_int8_per_tensor.json)|[Link](https://storage.openvinotoolkit.org/repositories/nncf/models/develop/squeezenet1_1_imagenet_int8_per_tensor.pth)|
|SqueezeNet V1.1|Mixed, 54.7% INT8 / 45.3% INT4|ImageNet|58.9|[squeezenet1_1_imagenet_mixed_int_manual.json](configs/mixed_precision/squeezenet1_1_imagenet_mixed_int_manual.json)|[Link](https://storage.openvinotoolkit.org/repositories/nncf/models/develop/squeezenet1_1_imagenet_int4_int8.pth)|


#### Binarization

As an example of NNCF convolution binarization capabilities, you may use the configs in `examples/torch/classification/configs/binarization` to binarize ResNet18. Use the same steps/command line parameters as for quantization (for best results, specify `--pretrained`), except for the actual binarization config path.

### Results for binarization
|Model|Compression algorithm|Dataset|PyTorch compressed accuracy|NNCF config file|PyTorch Checkpoint|
| :---: | :---: | :---: | :---: | :---: | :---: |
|ResNet-18|None|ImageNet|69.8|[resnet18_imagenet.json](configs/binarization/resnet18_imagenet.json)|-|
|ResNet-18|XNOR (weights), scale/threshold (activations)|ImageNet|61.63|[resnet18_imagenet_binarization_xnor.json](configs/binarization/resnet18_imagenet_binarization_xnor.json)|[Link](https://storage.openvinotoolkit.org/repositories/nncf/models/develop/resnet18_imagenet_binarization_xnor.pth)|
|ResNet-18|DoReFa (weights), scale/threshold (activations)|ImageNet|61.61|[resnet18_imagenet_binarization_dorefa.json](configs/binarization/resnet18_imagenet_binarization_dorefa.json)|[Link](https://storage.openvinotoolkit.org/repositories/nncf/models/develop/resnet18_imagenet_binarization_dorefa.pth)|


### Results for filter pruning
|Model|Compression algorithm|Dataset|PyTorch compressed accuracy|NNCF config file|PyTorch Checkpoint|
| :---: | :---: | :---: | :---: | :---: | :---: |
|ResNet-50|None|ImageNet|76.16|[resnet50_imagenet.json](configs/quantization/resnet50_imagenet.json)|-|
|ResNet-50|Filter pruning, 40%, geometric median criterion|ImageNet|75.62|[resnet50_pruning_geometric_median.json](configs/pruning/resnet50_pruning_geometric_median.json)|[Link](https://storage.openvinotoolkit.org/repositories/nncf/models/develop/resnet50_imagenet_filter_pruning_geomean.pth)|
|ResNet-18|None|ImageNet|69.8|[resnet18_imagenet.json](configs/binarization/resnet18_imagenet.json)|-|
|ResNet-18|Filter pruning, 40%, magnitude criterion|ImageNet|69.26|[resnet18_pruning_magnitude.json](configs/pruning/resnet18_pruning_magnitude.json)|[Link](https://storage.openvinotoolkit.org/repositories/nncf/models/develop/resnet18_imagenet_filter_pruning_magnitude.pth)|
|ResNet-18|Filter pruning, 40%, geometric median criterion|ImageNet|69.32|[resnet18_pruning_geometric_median.json](configs/pruning/resnet18_pruning_geometric_median.json)|[Link](https://storage.openvinotoolkit.org/repositories/nncf/models/develop/resnet18_imagenet_filter_pruning_geomean.pth)|
|ResNet-34|None|ImageNet|73.26|[resnet34_imagenet.json](configs/pruning/resnet34_imagenet.json)|-|
|ResNet-34|Filter pruning, 40%, geometric median criterion|ImageNet|72.72|[resnet34_pruning_geometric_median.json](configs/pruning/resnet34_pruning_geometric_median.json)|[Link](https://storage.openvinotoolkit.org/repositories/nncf/models/develop/resnet34_imagenet_filter_pruning_geomean.pth)|
|GoogLeNet|None|ImageNet|69.72|[googlenet_imagenet.json](configs/pruning/googlenet_imagenet.json)|-|
|GoogLeNet|Filter pruning, 40%, geometric median criterion|ImageNet|68.89|[googlenet_pruning_geometric_median.json](configs/pruning/googlenet_pruning_geometric_median.json)|[Link](https://storage.openvinotoolkit.org/repositories/nncf/models/develop/googlenet_imagenet_filter_pruning_geomean.pth)|
