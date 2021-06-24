# Neural Network Compression Framework (NNCF)

NNCF provides a suite of advanced compression algorithms, applied during a model fine-tuning and samples for models 
compression. It is highly aligned with [OpenVINO&trade;](https://github.com/openvinotoolkit/openvino) and is aimed 
to improve runtime performance of models in OpenVINO&trade; with minimal accuracy drop.

NNCF is designed to work with models from the two most popular deep learning frameworks, [PyTorch](https://pytorch.org/) 
and [TensorFlow](https://www.tensorflow.org/).

The framework is organized as a Python\* package that can be built and used in a standalone mode. The framework 
architecture is unified to make it easy to add different compression algorithms for both PyTorch and TensorFlow deep 
learning frameworks.

The samples demonstrate the usage of compression algorithms for three different use cases on public PyTorch and 
TensorFlow models and datasets: Image Classification, Object Detection and Semantic Segmentation. 
[Compression results](#nncf-compressed-model-zoo) achievable with the NNCF-powered samples can be found in a table at 
the end of this document.

## Key Features

- Support of various compression algorithms, applied during a model fine-tuning process to achieve best compression 
  parameters and accuracy:
  
  |Compression algorithm|PyTorch|TensorFlow|
  | :--- | :---: | :---: |
  |[Quantization](./docs/compression_algorithms/Quantization.md) | Supported | Supported |
  |[Mixed-Precision Quantization](./docs/compression_algorithms/Quantization.md#mixed_precision_quantization) | Supported | Support planned |
  |[Binarization](./docs/compression_algorithms/Binarization.md) | Supported | Not supported |
  |[Sparsity](./docs/compression_algorithms/Sparsity.md) | Supported | Supported |
  |[Filter pruning](./docs/compression_algorithms/Pruning.md) | Supported | Supported |

- Automatic, configurable model graph transformation to obtain the compressed model.
  > **NOTE**: Limited support for TensorFlow models. The models created using Sequential or Keras Functional API are only supported.
- Common interface for compression methods.
- GPU-accelerated layers for faster compressed model fine-tuning.
- Distributed training support.
- Configuration file examples for each supported compression algorithm.
- Git patches for prominent third-party repositories ([mmdetection](https://github.com/open-mmlab/mmdetection), [huggingface-transformers](https://github.com/huggingface/transformers)) demonstrating the process of integrating NNCF into custom training pipelines
- Exporting PyTorch compressed models to ONNX\* checkpoints and TensorFlow compressed models to SavedModel or Frozen Graph format, ready to use with [OpenVINO&trade; toolkit](https://github.com/openvinotoolkit/).

## Usage
The NNCF is organized as a regular Python package that can be imported in your target training pipeline script.
The basic workflow is loading a JSON configuration script containing NNCF-specific parameters determining the compression to be applied to your model, and then passing your model along with the configuration script to the `create_compressed_model` function.
This function returns a model with additional modifications necessary to enable algorithm-specific compression during fine-tuning and handle to the object allowing you to control the compression during the training process:

### Usage example with PyTorch 

```python
import torch
import nncf  # Important - should be imported directly after torch

from nncf import NNCFConfig
from nncf.torch import create_compressed_model, register_default_init_args

# Instantiate your uncompressed model
from torchvision.models.resnet import resnet50
model = resnet50()

# Load a configuration file to specify compression
nncf_config = NNCFConfig.from_json("resnet50_int8.json")

# Provide data loaders for compression algorithm initialization, if necessary
import torchvision.datasets as datasets
representative_dataset = datasets.ImageFolder("/path")
init_loader = torch.utils.data.DataLoader(representative_dataset)
nncf_config = register_default_init_args(nncf_config, init_loader)

# Apply the specified compression algorithms to the model
compression_ctrl, compressed_model = create_compressed_model(model, nncf_config)

# Now use compressed_model as a usual torch.nn.Module 
# to fine-tune compression parameters along with the model weights

# ... the rest of the usual PyTorch-powered training pipeline

# Export to ONNX or .pth when done fine-tuning
compression_ctrl.export_model("compressed_model.onnx")
torch.save(compressed_model.state_dict(), "compressed_model.pth")
```

### Usage example with TensorFlow
```python
import tensorflow as tf

from nncf import NNCFConfig
from nncf.tensorflow import create_compressed_model, register_default_init_args

# Instantiate your uncompressed model
from tensorflow.keras.applications import ResNet50
model = ResNet50()

# Load a configuration file to specify compression
nncf_config = NNCFConfig.from_json("resnet50_int8.json")

# Provide dataset for compression algorithm initialization
representative_dataset = tf.data.Dataset.list_files("/path/*.jpeg")
nncf_config = register_default_init_args(nncf_config, representative_dataset, batch_size=1)

# Apply the specified compression algorithms to the model
compression_ctrl, compressed_model = create_compressed_model(model, nncf_config)

# Now use compressed_model as a usual Keras model
# to fine-tune compression parameters along with the model weights

# ... the rest of the usual TensorFlow-powered training pipeline

# Export to Frozen Graph, TensorFlow SavedModel or .h5  when done fine-tuning 
compression_ctrl.export_model("compressed_model.pb", save_format='frozen_graph')
```

For a more detailed description of NNCF usage in your training code, see [this tutorial](docs/Usage.md). 
For in-depth examples of NNCF integration, browse the [sample scripts](#model-compression-samples) code, or the [example patches](#third-party-repository-integration) to third-party repositories.


## Model Compression Samples

For a quicker start with NNCF-powered compression, you can also try the sample scripts, each of which provides a basic training pipeline for classification, semantic segmentation and object detection neural network training correspondingly.

To run the samples please refer to the corresponding tutorials:

- PyTorch samples:
  - [Image Classification sample](examples/torch/classification/README.md)
  - [Object Detection sample](examples/torch/object_detection/README.md)
  - [Semantic Segmentation sample](examples/torch/semantic_segmentation/README.md)
- TensorFlow samples:
  - [Image Classification sample](examples/tensorflow/classification/README.md)
  - [Object Detection sample](examples/tensorflow/object_detection/README.md)
  - [Instance Segmentation sample](examples/tensorflow/semantic_segmentation/README.md)

## Third-party repository integration
NNCF may be straightforwardly integrated into training/evaluation pipelines of third-party repositories.
See [third_party_integration](./third_party_integration) for examples of code modifications (Git patches and base commit IDs are provided) that are necessary to integrate NNCF into select repositories.


## System requirements
- Ubuntu\* 18.04 or later (64-bit)
- Python\* 3.6.2 or later
- Supported frameworks:
  - PyTorch\* 1.5 or later (1.8.0 not supported, 1.8.1 supported)
  - TensorFlow\* 2.4.2

This repository is tested on Python* 3.6.2+, PyTorch* 1.8.1 (NVidia CUDA\* Toolkit 10.2) and TensorFlow* 2.4.2 (NVidia CUDA\* Toolkit 11.0).

## Installation
We suggest to install or use the package in the [Python virtual environment](https://docs.python.org/3/tutorial/venv.html).

First, create a virtual environment with the version of Python you're going to use and activate it.

Then you need to install one of the deep learning frameworks [PyTorch](https://pytorch.org/) or [TensorFlow](https://www.tensorflow.org/). Please follow to [TensorFlow installation guide](https://www.tensorflow.org/install/) or [PyTorch installation guide](https://pytorch.org/get-started/locally/#start-locally).

When one of those backends has been installed, NNCF can be installed as follow: 

#### As a package built from a checked-out repository:

Install the package and its dependencies by running the following in the repository root directory:
```
python setup.py install
```
Alternatively, If you don't install any backend you can install NNCF and PyTorch in one line with:
```
python setup.py install --torch
```
Install NNCF and TensorFlow in one line:
```
python setup.py install --tf
```

_NB_: For launching example scripts in this repository, we need to  recommend replacing the `install` option above with `develop` and setting the `PYTHONPATH` variable to the root of the checked-out repository.

#### As a PyPI package:

NNCF can be installed as a regular PyPI package via pip:
```
pip install nncf
```
Alternatively, If you don't install any backend you can install NNCF and PyTorch in one line with:
```
pip install nncf[torch]
```
Install NNCF and TensorFlow in one line:
```
pip install nncf[tf]
```

#### As a Docker image
Use one of the Dockerfiles in the [docker](./docker) directory to build an image with an environment already set up and ready for running NNCF [sample scripts](#model-compression-samples).

## Contributing
Refer to the [CONTRIBUTING.md](./CONTRIBUTING.md) file for guidelines on contributions to the NNCF repository.

## NNCF Compressed Model Zoo

Results achieved using sample scripts, example patches to third-party repositories and NNCF configuration files provided 
with this repository. See README.md files for [sample scripts](#model-compression-samples) and [example patches](#third-party-repository-integration) 
to find instruction and links to exact configuration files and final checkpoints.
- [PyTorch models](#pytorch-models)
  * [Classification](#pytorch_classification)
  * [Object detection](#pytorch_object_detection)
  * [Semantic segmentation](#pytorch_semantic_segmentation)
  * [Natural language processing (3rd-party training pipelines)](#pytorch_nlp)
  * [Object detection (3rd-party training pipelines)](#pytorch_object_detection_mmdetection)
  * [Instance Segmentation (3rd-party training pipelines)](#pytorch_instance_segmentation_mmdetection)
- [TensorFlow models](#tensorflow-models)
  * [Classification](#tensorflow_classification)
  * [Object detection](#tensorflow_object_detection)
  * [Instance segmentation](#tensorflow_instance_segmentation)  

### PyTorch models

<a name="pytorch_classification"></a>
#### Classification

|PyTorch Model|<img width="115" height="1">Compression algorithm<img width="115" height="1">|Dataset|Accuracy (Drop) %|
| :---: | :---: | :---: | :---: |
|ResNet-50|INT8|ImageNet|76.42 (-0.26)|
|ResNet-50|INT8 (per-tensor for weights)|ImageNet|76.37 (-0.21)|
|ResNet-50|Mixed, 44.8% INT8 / 55.2% INT4|ImageNet|76.2 (-0.04)|
|ResNet-50|INT8 + Sparsity 61% (RB)|ImageNet|75.43 (0.73)|
|ResNet-50|INT8 + Sparsity 50% (RB)|ImageNet|75.55 (0.61)|
|ResNet-50|Filter pruning, 40%, geometric median criterion|ImageNet|75.62 (0.54)|
|Inception V3|INT8|ImageNet|78.25 (-0.91)|
|Inception V3|INT8 + Sparsity 61% (RB)|ImageNet|77.58 (-0.24)|
|MobileNet V2|INT8|ImageNet|71.35 (0.58)|
|MobileNet V2|INT8 (per-tensor for weights)|ImageNet|71.3 (0.63)|
|MobileNet V2|Mixed, 46.6% INT8 / 53.4% INT4|ImageNet|70.92 (1.01)|
|MobileNet V2|INT8 + Sparsity 52% (RB)|ImageNet|71.11 (0.82)|
|SqueezeNet V1.1|INT8|ImageNet|58.28 (-0.04)|
|SqueezeNet V1.1|INT8 (per-tensor for weights)|ImageNet|58.26 (-0.02)|
|SqueezeNet V1.1|Mixed, 54.7% INT8 / 45.3% INT4|ImageNet|58.9 (-0.66)|
|ResNet-18|XNOR (weights), scale/threshold (activations)|ImageNet|61.63 (8.17)|
|ResNet-18|DoReFa (weights), scale/threshold (activations)|ImageNet|61.61 (8.19)|
|ResNet-18|Filter pruning, 40%, magnitude criterion|ImageNet|69.26 (0.54)|
|ResNet-18|Filter pruning, 40%, geometric median criterion|ImageNet|69.32 (0.48)|
|ResNet-34|Filter pruning, 40%, geometric median criterion|ImageNet|72.73 (0.57)|
|GoogLeNet|Filter pruning, 40%, geometric median criterion|ImageNet|68.82 (0.93)|

<a name="pytorch_object_detection"></a>
#### Object detection

|PyTorch Model|Compression algorithm|Dataset|mAP (drop) %|
| :---: | :---: | :---: | :---: |
|SSD300-MobileNet|INT8 + Sparsity 70% (Magnitude)|VOC12+07 train, VOC07 eval|62.94 (-0.71)|
|SSD300-VGG-BN|INT8|VOC12+07 train, VOC07 eval|77.96 (0.32)|
|SSD300-VGG-BN|INT8 + Sparsity 70% (Magnitude)|VOC12+07 train, VOC07 eval|77.59 (0.69)|
|SSD300-VGG-BN|Filter pruning, 40%, geometric median criterion|VOC12+07 train, VOC07 eval|77.72 (0.56)|
|SSD512-VGG-BN|INT8|VOC12+07 train, VOC07 eval|80.12 (0.14)|
|SSD512-VGG-BN|INT8 + Sparsity 70% (Magnitude)|VOC12+07 train, VOC07 eval|79.67 (0.59)|

<a name="pytorch_semantic_segmentation"></a>
#### Semantic segmentation

|PyTorch Model|<img width="125" height="1">Compression algorithm<img width="125" height="1">|Dataset|Accuracy (Drop) %|
| :---: | :---: | :---: | :---: |
|UNet|INT8|CamVid|71.8 (0.15)|
|UNet|INT8 + Sparsity 60% (Magnitude)|CamVid|72.03 (-0.08)|
|ICNet|INT8|CamVid|67.86 (0.03)|
|ICNet|INT8 + Sparsity 60% (Magnitude)|CamVid|67.18 (0.71)|
|UNet|INT8|Mapillary|55.87 (0.36)|
|UNet|INT8 + Sparsity 60% (Magnitude)|Mapillary|55.65 (0.58)|
|UNet|Filter pruning, 25%, geometric median criterion|Mapillary|55.62 (0.61)|

<a name="pytorch_nlp"></a>
#### NLP

|PyTorch Model|<img width="20" height="1">Compression algorithm<img width="20" height="1">|Dataset|Accuracy (Drop) %|
| :---: | :---: | :---: | :---: |
|BERT-base-chinese|INT8|XNLI|77.22 (0.46)|
|BERT-large (Whole Word Masking)|INT8|SQuAD v1.1|F1: 92.68 (0.53)|
|RoBERTa-large|INT8|MNLI|matched: 89.25 (1.35)|
|DistilBERT-base|INT8|SST-2|90.3 (0.8)|
|MobileBERT|INT8|SQuAD v1.1|F1: 89.4 (0.58)|
|GPT-2|INT8|WikiText-2 (raw)|perplexity: 20.9 (-1.17)|

<a name="pytorch_object_detection_mmdetection"></a>
#### Object detection (MMDetection)

|PyTorch Model|<img width="60" height="1">Compression algorithm<img width="60" height="1">|Dataset|mAP (drop) %|
| :---: | :---: | :---: | :---: |
|RetinaNet-ResNet50-FPN|INT8|COCO2017|bbox: 35.3 (0.3)|
|RetinaNet-ResNet50-FPN|INT8 + Sparsity 50%|COCO2017|bbox: 34.7 (0.9)|
|RetinaNet-ResNeXt101-64x4d-FPN|INT8|COCO2017|bbox: 39.1 (0.5)|

<a name="pytorch_instance_segmentation_mmdetection"></a>
#### Instance Segmentation (MMDetection)

|PyTorch Model|<img width="85" height="1">Compression algorithm<img width="85" height="1">|Dataset|mAP (drop) %|
| :---: | :---: | :---: | :---: |
|Mask-RCNN-ResNet50-FPN|INT8|COCO2017|bbox: 40.6 (0.2)<br/>segm: 36.5 (0.5)|

### TensorFlow models

<a name="tensorflow_classification"></a>
#### Classification

|Tensorflow Model|Compression algorithm|Dataset|Accuracy (Drop) %|
| :---: | :---: | :---: | :---: |
|Inception V3|INT8 (per-tensor for weights)|ImageNet|78.35 (-0.45)|
|Inception V3|Sparsity 54% (Magnitude)|ImageNet|77.87 (0.03)|
|Inception V3|INT8 (per-tensor for weights) + Sparsity 61% (RB)|ImageNet|77.58 (0.32)|
|MobileNet V2|INT8 (per-tensor for weights)|ImageNet|71.66 (0.19)|
|MobileNet V2|Sparsity 50% (RB)|ImageNet|71.34 (0.51)|
|MobileNet V2|INT8 (per-tensor for weights) + Sparsity 52% (RB)|ImageNet|71.0 (0.85)|
|MobileNet V3 small|INT8 (per-channel, symmetric for weights; per-tensor, asymmetric for activations) |ImageNet|67.7 (0.68)|
|MobileNet V3 small|INT8 (per-channel, symmetric for weights; per-tensor, asymmetric for activations) + Sparsity 42% (RB)|ImageNet|67.7 (0.68)|
|MobileNet V3 large|INT8 (per-channel, symmetric for weights; per-tensor, asymmetric for activations) |ImageNet|75.0 (0.81)|
|MobileNet V3 large|INT8 (per-channel, symmetric for weights; per-tensor, asymmetric for activations) + Sparsity 42% (RB)|ImageNet|75.15 (0.66)|
|ResNet50|INT8 (per-tensor for weights)|ImageNet|75.0 (0.04)|
|ResNet50|Sparsity 80% (RB)|ImageNet|74.36 (0.68)|
|ResNet50|INT8 (per-tensor for weightsy) + Sparsity 65% (RB)|ImageNet|74.3 (0.74)|
|ResNet50|Filter Pruning 40%, geometric_median criterion|ImageNet|74.98 (0.06)|
|ResNet50|Filter Pruning 40%, geometric_median criterion + INT8 (per-tensor for weights)|ImageNet|75.08 (-0.04)|
|TensorFlow Hub MobileNet V2|Sparsity 35% (Magnitude)|ImageNet|71.90 (-0.06)|

<a name="tensorflow_object_detection"></a>
#### Object detection

|TensorFlow Model|Compression algorithm|Dataset|mAP (drop) %|
| :---: | :---: | :---: | :---: |
|RetinaNet|INT8 (per-tensor for weights)|COCO2017|33.22 (0.22)|
|RetinaNet|Sparsity 50% (Magnitude)|COCO2017|33.13 (0.31)|
|RetinaNet|Filter Pruning 40%, geometric_median criterion|COCO2017|32.7 (0.74)|
|RetinaNet|Filter Pruning 40%, geometric_median criterion + INT8 (per-tensor for weights)|COCO2017|32.82 (0.62)|
|YOLOv4|INT8 (per-channel, symmetric for weights; per-tensor, asymmetric for activations)|COCO2017|46.15 (0.89)|
|YOLOv4|Sparsity 50% (Magnitude)|COCO2017|46.54 (0.50)|

<a name="tensorflow_instance_segmentation"></a>
#### Instance segmentation

|TensorFlow Model|<img width="110" height="1">Compression algorithm<img width="110" height="1">|Dataset|mAP (drop) %|
| :---: | :---: | :---: | :---: |
|MaskRCNN|INT8 (per-tensor for weights)|COCO2017|bbox: 37.12 (0.21)<br/>segm: 33.52 (0.04)|
|MaskRCNN|Sparsity 50% (Magnitude)|COCO2017|bbox: 36.93 (0.40)<br/>segm: 33.23 (0.33)|

## Citing

```
@article{kozlov2020neural,
    title =   {Neural network compression framework for fast model inference},
    author =  {Kozlov, Alexander and Lazarevich, Ivan and Shamporov, Vasily and Lyalyushkin, Nikolay and Gorbachev, Yury},
    journal = {arXiv preprint arXiv:2002.08679},
    year =    {2020}
}
```

## Legal Information
[*] Other names and brands may be claimed as the property of others.
