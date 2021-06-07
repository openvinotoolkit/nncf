---
> #### This directory contains the experimental implementation of the Neural Network Compression Framework for TensorFlow. The implementation is subject to change without notice and offers no guarantees, including for the public API. Stay tuned.

---

# Neural Network Compression Framework for TensorFlow (NNCF TF)

This repository contains a TensorFlow*-based framework and samples for neural networks compression.

The framework is the implementaiton of the [Neural Network Compression Framework (NNCF)](https://github.com/openvinotoolkit/nncf_pytorch) for TensorFlow\*.

The framework is organized as a Python\* package that can be built and used in a standalone mode.
The framework architecture is unified to make it easy to add different compression methods.
 
The samples demonstrate the usage of compression algorithms for two different use cases on public models and datasets: Image Classification, Object Detection.
[Compression results](#nncf-tf-compression-results) achievable with the NNCF-powered samples can be found in a table at the end of this document.

## Key Features

- Support of various compression algorithms, applied during a model fine-tuning process to achieve best compression parameters and accuracy:
    - Quantization
    - Sparsity
- Automatic, configurable model graph transformation to obtain the compressed model. The model is wrapped by the custom class and additional compression-specific layers are inserted in the graph.
  > **NOTE**: Only Keras models created using Sequential or Keras Functional API are supported.
- Common interface for compression methods.
- Distributed training support.
- Configuration file examples for each supported compression algorithm.
- Exporting compressed models to Frozen Graph or TensorFlow\* SavedModel ready for usage with [OpenVINO&trade; toolkit](https://github.com/openvinotoolkit/).

## Usage
The NNCF TF is organized as a regular Python package that can be imported in an arbitrary training script.
The basic workflow is loading a JSON configuration script containing NNCF-specific parameters determining the compression to be applied to your model, and then passing your model along with the configuration script to the `nncf.tensorflow.create_compressed_model` function.
This function returns a transformed model ready for compression fine-tuning, and handle to the object allowing you to control the compression during the training process:

```python
import tensorflow as tf

from nncf import NNCFConfig
from nncf.tensorflow import create_compressed_model
from nncf.tensorflow.initialization import register_default_init_args

# Instantiate your uncompressed model
from tensorflow.keras.applications import ResNet50
model = ResNet50()

# Load a configuration file to specify compression
nncf_config = NNCFConfig.from_json("resnet50_imagenet_int8.json")

# Provide dataset for compression algorithm initialization
dataset = tf.data.Dataset.list_files("/path/*.txt")
nncf_config = register_default_init_args(nncf_config, dataset, batch_size=1)

# Apply the specified compression algorithms to the model
compression_ctrl, compressed_model = create_compressed_model(model, nncf_config)

# Now use compressed_model as a usual Keras model

# ... the rest of the usual TensorFlow-powered training pipeline

# Export to Frozen Graph, TensorFlow SavedModel or .h5  when done fine-tuning 
compression_ctrl.export_model("compressed_model.pb", save_format='frozen_graph')
```

## Model Compression Samples

For a quick start with NNCF-powered compression, you can also try the sample scripts, each of them provides a basic training pipeline for Image Classification and Object Detection correspondingly.

To run the samples please refer to the corresponding tutorials:
- [Image Classification sample](examples/tensorflow/classification/README.md)
- [Object Detection sample](examples/tensorflow/object_detection/README.md)
- [Instance Segmentation sample](examples/tensorflow/segmentation/README.md)

## System requirements
- Ubuntu\* 16.04 or later (64-bit)
- Python\* 3.6.2 or later
- NVidia CUDA\* Toolkit 11.0
- TensorFlow\* 2.4.0

## NNCF TF compression results

Achieved using sample scripts and NNCF TF configuration files provided with this repository. See README files for [sample scripts](#model-compression-samples) for links to exact configuration files and pre-trained models.

Quick jump to the samples:
- [Classification](#classification)
- [Object Detection](#object-detection)
- [Instance Segmentation](#instance-segmentation)

#### Classification

|**Model**|**Compression algorithm**|**Dataset**|**TensorFlow FP32 baseline**|**TensorFlow compressed accuracy**|
| :---: | :---: | :---: | :---: | :---: |
|Inception V3|INT8 w:sym,per-tensor a:sym,per-tensor |ImageNet|77.9|78.35|
|Inception V3|Sparsity 54% (Magnitude)|ImageNet|77.9|77.87|
|Inception V3|INT8 w:sym,per-tensor a:sym,per-tensor + Sparsity 61% (RB)|ImageNet|77.9|77.58|
|MobileNet V2|INT8 w:sym,per-tensor a:sym,per-tensor |ImageNet|71.85|71.66|
|MobileNet V2|Sparsity 50% (RB)|ImageNet|71.85|71.34|
|MobileNet V2|INT8 w:sym,per-tensor a:sym,per-tensor + Sparsity 52% (RB)|ImageNet|71.85|71.0|
|MobileNet V3 small|INT8 w:sym,per-channel a:asym,per-tensor |ImageNet|68.38|67.7|
|MobileNet V3 small|INT8 w:sym,per-channel a:asym,per-tensor + Sparsity 42% (RB)|ImageNet|68.38|67.7|
|MobileNet V3 large|INT8 w:sym,per-channel a:asym,per-tensor |ImageNet|75.81|75.0|
|MobileNet V3 large|INT8 w:sym,per-channel a:asym,per-tensor + Sparsity 42% (RB)|ImageNet|75.81|75.15|
|ResNet50|INT8 w:sym,per-tensor a:sym,per-tensor|ImageNet|75.04|75.0|
|ResNet50|Sparsity 80% (RB)|ImageNet|75.04|74.36|
|ResNet50|INT8 w:sym,per-tensor a:sym,per-tensor + Sparsity 65% (RB)|ImageNet|75.04|74.3|
|ResNet50|Filter Pruning 40%|ImageNet|75.04|74.98|
|TensorFlow Hub MobileNet V2|Sparsity 35% (Magnitude)|ImageNet|71.84|71.90|

#### Object detection

|**Model**|**Compression algorithm**|**Dataset**|**TensorFlow FP32 baseline mAP**|**TensorFlow compressed mAP**|
| :---: | :---: | :---: | :---: | :---: |
|RetinaNet|INT8 w:sym,per-tensor a:sym,per-tensor |COCO2017|33.44|33.26|
|RetinaNet|Sparsity 50% (Magnitude)|COCO2017|33.44|33.13|
|RetinaNet|Filter Pruning 40%|COCO2017|33.44|32.7|

#### Instance Segmentation

|**Model**|**Compression algorithm**|**Dataset**|**TensorFlow FP32 baseline mAP**|**TensorFlow compressed mAP**|
| :---: | :---: | :---: | :---: | :---: |
|MaskRCNN|INT8 w:sym,per-tensor a:sym,per-tensor |COCO2017|bbox: 37.33<br/>segm: 33.56|bbox: 37.25<br/>segm: 33.59|
|MaskRCNN|Sparsity 50% (Magnitude)|COCO2017|bbox: 37.33<br/>segm: 33.56|bbox: 36.93<br/>segm: 33.23|

## Legal Information
[*] Other names and brands may be claimed as the property of others.
