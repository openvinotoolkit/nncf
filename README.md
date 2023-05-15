<div align="center">

# Neural Network Compression Framework (NNCF)

[![GitHub Release](https://img.shields.io/github/v/release/openvinotoolkit/nncf?color=green)](https://github.com/openvinotoolkit/nncf/releases)
[![Website](https://img.shields.io/website?up_color=blue&up_message=docs&url=https%3A%2F%2Fdocs.openvino.ai%2Flatest%2Fopenvino_docs_model_optimization_guide.html)](https://docs.openvino.ai/latest/openvino_docs_model_optimization_guide.html)
[![Apache License Version 2.0](https://img.shields.io/badge/license-Apache_2.0-green.svg)](LICENSE)
[![PyPI Downloads](https://static.pepy.tech/badge/nncf)](https://pypi.org/project/nncf/)
 
</div>

_For the installation instructions, [click here](#installation)._

NNCF provides a suite of advanced algorithms for Neural Networks inference optimization in [OpenVINO&trade;](https://docs.openvino.ai/latest/home.html) with minimal accuracy drop.

NNCF is designed to work with models from [PyTorch](https://pytorch.org/), [TensorFlow](https://www.tensorflow.org/), [ONNX](https://onnx.ai/) and [OpenVINO&trade;](https://docs.openvino.ai/latest/home.html).

NNCF provides samples that demonstrate the usage of compression algorithms for three different use cases on public PyTorch and 
TensorFlow models and datasets: Image Classification, Object Detection and Semantic Segmentation. 
[Compression results](#nncf-compressed-model-zoo) achievable with the NNCF-powered samples can be found in a table at 
the end of this document.

The framework is organized as a Python\* package that can be built and used in a standalone mode. The framework 
architecture is unified to make it easy to add different compression algorithms for both PyTorch and TensorFlow deep 
learning frameworks.

## Key Features
### Post-Training Compression Algorithms

| Compression algorithm                                                       |PyTorch|TensorFlow|   ONNX   |       OpenVINO     |
|:----------------------------------------------------------------------------| :---: | :---: |:--------:|:------------------:|
| [Quantization](./docs/compression_algorithms/post_training/Quantization.md) | Supported | Supported |Supported| Preview |

_Preview means that this is a work in progress and NNCF does not guarantee the full functional support._

### Training-Time Compression Algorithms

|Compression algorithm|PyTorch|TensorFlow|
| :--- | :---: | :---: |
|[Quantization](./docs/compression_algorithms/Quantization.md) | Supported | Supported |
|[Mixed-Precision Quantization](./docs/compression_algorithms/Quantization.md#mixed_precision_quantization) | Supported | Not supported |
|[Binarization](./docs/compression_algorithms/Binarization.md) | Supported | Not supported |
|[Sparsity](./docs/compression_algorithms/Sparsity.md) | Supported | Supported |
|[Filter pruning](./docs/compression_algorithms/Pruning.md) | Supported | Supported |
|[Movement pruning](./nncf/experimental/torch/sparsity/movement/MovementSparsity.md) | Experimental | Not supported |





- Automatic, configurable model graph transformation to obtain the compressed model.
  > **NOTE**: Limited support for TensorFlow models. The models created using Sequential or Keras Functional API are only supported.
- Common interface for compression methods.
- GPU-accelerated layers for faster compressed model fine-tuning.
- Distributed training support.
- Configuration file examples for each supported compression algorithm.
- Git patch for prominent third-party repository ([huggingface-transformers](https://github.com/huggingface/transformers)) demonstrating the process of integrating NNCF into custom training pipelines
- Seamless combination of pruning, sparsity and quantization algorithms. Please refer [optimum-intel](https://github.com/huggingface/optimum-intel/tree/main/examples/openvino) for examples of 
joint (movement) pruning, quantization and distillation (JPQD), end-to-end from NNCF optimization to compressed OpenVINO IR.
- Exporting PyTorch compressed models to ONNX\* checkpoints and TensorFlow compressed models to SavedModel or Frozen Graph format, ready to use with [OpenVINO&trade; toolkit](https://docs.openvino.ai/latest/home.html).
- Support for [Accuracy-Aware model training](./docs/Usage.md#accuracy-aware-model-training) pipelines via the [Adaptive Compression Level Training](./docs/accuracy_aware_model_training/AdaptiveCompressionLevelTraining.md) and [Early Exit Training](./docs/accuracy_aware_model_training/EarlyExitTraining.md).

## Usage

### Post-Training Quantization

The NNCF PTQ is the simplest way to apply 8-bit quantization. To run the algorithm you only need your model and a small (~300 samples) calibration dataset.

[OpenVINO](https://github.com/openvinotoolkit/openvino) is the preferred backend to run PTQ with, and PyTorch, TensorFlow and ONNX are also supported.

<details open><summary><b>OpenVINO</b></summary>

```python
import nncf
import openvino.runtime as ov
import torch
from torchvision import datasets

# Instantiate your uncompressed model
model = ov.Core().read_model("/model_path")
# Provide validation part of the dataset to collect statistics needed for the compression algorithm
val_dataset = datasets.ImageFolder("/path")
dataset_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1)

# Step 1: Initialize transformation function
def transform_fn(data_item):
    images, _ = data_item
    return images

# Step 2: Initialize NNCF Dataset
calibration_dataset = nncf.Dataset(dataset_loader, transform_fn)
# Step 3: Run the quantization pipeline
quantized_model = nncf.quantize(model, calibration_dataset)
```

</details>

<details><summary><b>PyTorch</b></summary>

```python
import nncf
import torch
from torchvision import datasets, models

# Instantiate your uncompressed model
model = models.mobilenet_v2() 
# Provide validation part of the dataset to collect statistics needed for the compression algorithm
val_dataset = datasets.ImageFolder("/path")
dataset_loader = torch.utils.data.DataLoader(val_dataset)

# Step 1: Initialize the transformation function
def transform_fn(data_item):
    images, _ = data_item
    return images

# Step 2: Initialize NNCF Dataset
calibration_dataset = nncf.Dataset(dataset_loader, transform_fn)
# Step 3: Run the quantization pipeline
quantized_model = nncf.quantize(model, calibration_dataset)

```

</details>

<details><summary><b>TensorFlow</b></summary>

```python
import nncf
import tensorflow as tf
import tensorflow_datasets as tfds

# Instantiate your uncompressed model
model = tf.keras.applications.MobileNetV2()
# Provide validation part of the dataset to collect statistics needed for the compression algorithm
val_dataset = tfds.load("/path", split="validation", 
                        shuffle_files=False, as_supervised=True)

# Step 1: Initialize transformation function
def transform_fn(data_item):
    images, _ = data_item
    return images

# Step 2: Initialize NNCF Dataset
calibration_dataset = nncf.Dataset(val_dataset, transform_fn)
# Step 3: Run the quantization pipeline
quantized_model = nncf.quantize(model, calibration_dataset)
```

</details>

<details><summary><b>ONNX</b></summary>

```python
import onnx
import nncf
import torch
from torchvision import datasets

# Instantiate your uncompressed model
onnx_model = onnx.load_model("/model_path")
# Provide validation part of the dataset to collect statistics needed for the compression algorithm
val_dataset = datasets.ImageFolder("/path")
dataset_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1)

# Step 1: Initialize transformation function
input_name = onnx_model.graph.input[0].name
def transform_fn(data_item):
    images, _ = data_item
    return {input_name: images.numpy()}

# Step 2: Initialize NNCF Dataset
calibration_dataset = nncf.Dataset(dataset_loader, transform_fn)
# Step 3: Run the quantization pipeline
quantized_model = nncf.quantize(onnx_model, calibration_dataset)
```

</details>


[//]: # (NNCF provides full  [samples]&#40;#post-training-quantization-samples&#41;, which demonstrate Post-Training Quantization usage for PyTorch, TensorFlow, ONNX, OpenVINO.)

### Training-Time Compression

Below is an example of Accuracy Aware Quantization pipeline where model weights and compression parameters may be fine-tuned to achieve a higher accuracy.

<details><summary><b>PyTorch</b></summary>

```python
import torch
import nncf.torch  # Important - must be imported before any other external package that depends on torch

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

**NOTE (PyTorch)**: Due to the way NNCF works within the PyTorch backend, `import nncf` must be done before any other import of `torch` in your package _or_ in third-party packages that your code utilizes, otherwise the compression may be applied incompletely.

</details>

<details><summary><b>Tensorflow</b></summary>

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
compression_ctrl.export_model("compressed_model.pb", save_format="frozen_graph")
```

</details>

For a more detailed description of NNCF usage in your training code, see [this tutorial](docs/Usage.md).

## Model Compression Samples

For a quicker start with NNCF-powered compression, try sample notebooks and scripts presented below.

### Model Compression Notebooks 

A collection of ready-to-run Jupyter* notebooks are available to demonstrate how to use NNCF compression algorithms to optimize models for inference with the OpenVINO Toolkit:
- [Post-Training Quantization of Pytorch model with NNCF](https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/112-pytorch-post-training-quantization-nncf)
- [Optimizing PyTorch models with NNCF of OpenVINO by 8-bit quantization](https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/302-pytorch-quantization-aware-training)
- [Optimizing TensorFlow models with NNCF of OpenVINO by 8-bit quantization](https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/305-tensorflow-quantization-aware-training)

### Post-Training Quantization Samples
Compact scripts demonstrating quantization and corresponding inference speed boost: 
- [PyTorch Post-Training Quantization sample](examples/post_training_quantization/torch/mobilenet_v2/README.md)
- [TensorFlow Post-Training Quantization sample](examples/post_training_quantization/tensorflow/mobilenet_v2/README.md)
- [ONNX Post-Training Quantization sample](examples/post_training_quantization/onnx/mobilenet_v2/README.md)
- [OpenVINO Post-Training Quantization sample](examples/post_training_quantization/openvino/mobilenet_v2/README.md)

### Training-Time Compression Samples
These examples provide full pipelines including compression, training and inference for classification, object detection and segmentation tasks.
- PyTorch samples:
  - [Image Classification sample](examples/torch/classification/README.md)
  - [Object Detection sample](examples/torch/object_detection/README.md)
  - [Semantic Segmentation sample](examples/torch/semantic_segmentation/README.md)
- TensorFlow samples:
  - [Image Classification sample](examples/tensorflow/classification/README.md)
  - [Object Detection sample](examples/tensorflow/object_detection/README.md)
  - [Instance Segmentation sample](examples/tensorflow/segmentation/README.md)

## Third-party repository integration
NNCF may be straightforwardly integrated into training/evaluation pipelines of third-party repositories.

### Used by

- [OpenVINO Training Extensions](https://github.com/openvinotoolkit/training_extensions)
  
  NNCF is integrated into OpenVINO Training Extensions as model optimization backend. So you can train, optimize and export new models based on the available model templates as well as run exported models with OpenVINO.

### Git patches for third-party repository
See [third_party_integration](./third_party_integration) for examples of code modifications (Git patches and base commit IDs are provided) that are necessary to integrate NNCF into the following repositories:
  - [huggingface-transformers](third_party_integration/huggingface_transformers/README.md)

## System requirements
- Ubuntu\* 18.04 or later (64-bit)
- Python\* 3.7 or later
- Supported frameworks:
  - PyTorch\* 1.12.1
  - TensorFlow\* >=2.4.0, <=2.11.1

This repository is tested on Python* 3.8.10, PyTorch* 1.12.1 (NVidia CUDA\* Toolkit 11.6) and TensorFlow* 2.11.1 (NVidia CUDA\* Toolkit 11.2).

## Installation
For detailed installation instructions please refer to the [Installation](./docs/Installation.md) page.

NNCF can be installed as a regular PyPI package via pip:
```
pip install nncf
```
If you want to install both NNCF and the supported PyTorch version in one line, you can do this by simply running:
```
pip install nncf[torch]
```
Other viable options besides `[torch]` are `[tf]`, `[onnx]` and `[openvino]`.

NNCF is also available via [conda](https://anaconda.org/conda-forge/nncf):
```
conda install -c conda-forge nncf
```

You may also use one of the Dockerfiles in the [docker](./docker) directory to build an image with an environment already set up and ready for running NNCF [sample scripts](#model-compression-samples).

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
- [TensorFlow models](#tensorflow-models)
  * [Classification](#tensorflow_classification)
  * [Object detection](#tensorflow_object_detection)
  * [Instance segmentation](#tensorflow_instance_segmentation)

### PyTorch models

<a name="pytorch_classification"></a>
#### Classification

|Model|Compression algorithm|Dataset|Accuracy (_drop_) %|
| :---: | :---: | :---: | :---: |
|ResNet-50|INT8|ImageNet|76.46 (-0.31)|
|ResNet-50|INT8 (per-tensor only)|ImageNet|76.39 (-0.24)|
|ResNet-50|Mixed, 43.12% INT8 / 56.88% INT4|ImageNet|76.05 (0.10)|
|ResNet-50|INT8 + Sparsity 61% (RB)|ImageNet|75.42 (0.73)|
|ResNet-50|INT8 + Sparsity 50% (RB)|ImageNet|75.50 (0.65)|
|ResNet-50|Filter pruning, 40%, geometric median criterion|ImageNet|75.57 (0.58)|
|Inception V3|INT8|ImageNet|77.45 (-0.12)|
|Inception V3|INT8 + Sparsity 61% (RB)|ImageNet|76.36 (0.97)|
|MobileNet V2|INT8|ImageNet|71.07 (0.80)|
|MobileNet V2|INT8 (per-tensor only)|ImageNet|71.24 (0.63)|
|MobileNet V2|Mixed, 58.88% INT8 / 41.12% INT4|ImageNet|70.95 (0.92)|
|MobileNet V2|INT8 + Sparsity 52% (RB)|ImageNet|71.09 (0.78)|
|MobileNet V3 small|INT8|ImageNet|66.98 (0.68)|
|SqueezeNet V1.1|INT8|ImageNet|58.22 (-0.03)|
|SqueezeNet V1.1|INT8 (per-tensor only)|ImageNet|58.11 (0.08)|
|SqueezeNet V1.1|Mixed, 52.83% INT8 / 47.17% INT4|ImageNet|57.57 (0.62)|
|ResNet-18|XNOR (weights), scale/threshold (activations)|ImageNet|61.67 (8.09)|
|ResNet-18|DoReFa (weights), scale/threshold (activations)|ImageNet|61.63 (8.13)|
|ResNet-18|Filter pruning, 40%, magnitude criterion|ImageNet|69.27 (0.49)|
|ResNet-18|Filter pruning, 40%, geometric median criterion|ImageNet|69.31 (0.45)|
|ResNet-34|Filter pruning, 50%, geometric median criterion + KD|ImageNet|73.11 (0.19)|
|GoogLeNet|Filter pruning, 40%, geometric median criterion|ImageNet|69.47 (0.30)|

<a name="pytorch_object_detection"></a>
#### Object detection

|Model|Compression algorithm|Dataset|mAP (_drop_) %|
| :---: | :---: | :---: | :---: |
|SSD300-MobileNet|INT8 + Sparsity 70% (Magnitude)|VOC12+07 train, VOC07 eval|62.95 (-0.72)|
|SSD300-VGG-BN|INT8|VOC12+07 train, VOC07 eval|77.81 (0.47)|
|SSD300-VGG-BN|INT8 + Sparsity 70% (Magnitude)|VOC12+07 train, VOC07 eval|77.66 (0.62)|
|SSD300-VGG-BN|Filter pruning, 40%, geometric median criterion|VOC12+07 train, VOC07 eval|78.35 (-0.07)|
|SSD512-VGG-BN|INT8|VOC12+07 train, VOC07 eval|80.04 (0.22)|
|SSD512-VGG-BN|INT8 + Sparsity 70% (Magnitude)|VOC12+07 train, VOC07 eval|79.68 (0.58)|

<a name="pytorch_semantic_segmentation"></a>
#### Semantic segmentation

|Model|Compression algorithm|Dataset|mIoU (_drop_) %|
| :---: | :---: | :---: | :---: |
|UNet|INT8|CamVid|71.89 (0.06)|
|UNet|INT8 + Sparsity 60% (Magnitude)|CamVid|72.46 (-0.51)|
|ICNet|INT8|CamVid|67.89 (0.00)|
|ICNet|INT8 + Sparsity 60% (Magnitude)|CamVid|67.16 (0.73)|
|UNet|INT8|Mapillary|56.09 (0.15)|
|UNet|INT8 + Sparsity 60% (Magnitude)|Mapillary|55.69 (0.55)|
|UNet|Filter pruning, 25%, geometric median criterion|Mapillary|55.64 (0.60)|

<a name="pytorch_nlp"></a>
#### NLP (HuggingFace Transformers-powered models)

|PyTorch Model|<img width="20" height="1">Compression algorithm<img width="20" height="1">|Dataset|Accuracy (Drop) %|
| :---: | :---: | :---: | :---: |
|BERT-base-chinese|INT8|XNLI|77.22 (0.46)|
|BERT-base-cased|INT8|CoNLL2003|99.18 (-0.01)|
|BERT-base-cased|INT8|MRPC|84.8 (-0.24)|
|BERT-large (Whole Word Masking)|INT8|SQuAD v1.1|F1: 92.68 (0.53)|
|RoBERTa-large|INT8|MNLI|matched: 89.25 (1.35)|
|DistilBERT-base|INT8|SST-2|90.3 (0.8)|
|MobileBERT|INT8|SQuAD v1.1|F1: 89.4 (0.58)|
|GPT-2|INT8|WikiText-2 (raw)|perplexity: 20.9 (-1.17)|

### TensorFlow models

<a name="tensorflow_classification"></a>
#### Classification

|Model|Compression algorithm|Dataset|Accuracy (_drop_) %|
| :---: | :---: | :---: | :---: |
|Inception V3|INT8 (per-tensor symmetric for weights, per-tensor asymmetric half-range for activations)|ImageNet|78.39 (-0.48)|
|Inception V3|INT8 (per-tensor symmetric for weights, per-tensor asymmetric half-range for activations), Sparsity 61% (RB)|ImageNet|77.52 (0.39)|
|Inception V3|Sparsity 54% (Magnitude)|ImageNet|77.86 (0.05)|
|MobileNet V2|INT8 (per-tensor symmetric for weights, per-tensor asymmetric half-range for activations)|ImageNet|71.63 (0.22)|
|MobileNet V2|INT8 (per-tensor symmetric for weights, per-tensor asymmetric half-range for activations), Sparsity 52% (RB)|ImageNet|70.94 (0.91)|
|MobileNet V2| Sparsity 50% (RB)|ImageNet|71.34 (0.51)|
|MobileNet V2 (TensorFlow Hub MobileNet V2)|Sparsity 35% (Magnitude)|ImageNet|71.87 (-0.02)|
|MobileNet V3 (Small)|INT8 (per-channel symmetric for weights, per-tensor asymmetric half-range for activations)|ImageNet|67.79 (0.59)|
|MobileNet V3 (Small)|INT8 (per-channel symmetric for weights, per-tensor asymmetric half-range for activations) + Sparsity 42% (Magnitude)|ImageNet|67.44 (0.94)|
|MobileNet V3 (Large)|INT8 (per-channel symmetric for weights, per-tensor asymmetric half-range for activations)|ImageNet|75.04 (0.76)|
|MobileNet V3 (Large)|INT8 (per-channel symmetric for weights, per-tensor asymmetric half-range for activations) + Sparsity 42% (RB)|ImageNet|75.24 (0.56)|
|ResNet-50|INT8|ImageNet|74.99 (0.06)|
|ResNet-50|INT8 (per-tensor symmetric for weights, per-tensor asymmetric half-range for activations) + Sparsity 65% (RB)|ImageNet|74.36 (0.69)|
|ResNet-50|Sparsity 80% (RB)|ImageNet|74.38 (0.67)|
|ResNet-50|Filter pruning, 40%, geometric median criterion|ImageNet|74.96 (0.09)|
|ResNet-50|INT8 (per-tensor symmetric for weights, per-tensor asymmetric half-range for activations) + Filter pruning, 40%, geometric median criterion|ImageNet|75.09 (-0.04)|

<a name="tensorflow_object_detection"></a>
#### Object detection

|Model|Compression algorithm|Dataset|mAP (_drop_) %|
| :---: | :---: | :---: | :---: |
|RetinaNet|INT8 (per-tensor symmetric for weights, per-tensor asymmetric half-range for activations)|COCO 2017|33.12 (0.31)|
|RetinaNet|Magnitude sparsity (50%)|COCO 2017|33.10 (0.33)|
|RetinaNet|Filter pruning, 40%|COCO 2017|32.72 (0.71)|
|RetinaNet|INT8 (per-tensor symmetric for weights, per-tensor asymmetric half-range for activations) + filter pruning 40%|COCO 2017|32.67 (0.76)|
|YOLO v4|INT8 (per-channel symmetric for weights, per-tensor asymmetric half-range for activations)|COCO 2017|46.20 (0.87)|
|YOLO v4|Magnitude sparsity, 50%|COCO 2017|46.49 (0.58)|

<a name="tensorflow_instance_segmentation"></a>
#### Instance segmentation

|Model|Compression algorithm|Dataset|mAP (_drop_) %|
| :---: | :---: | :---: | :---: |
|Mask-R-CNN|INT8 (per-tensor symmetric for weights, per-tensor asymmetric half-range for activations)|COCO 2017|37.19 (0.14)|
|Mask-R-CNN|Magnitude sparsity, 50%|COCO 2017|36.94 (0.39)|

### ONNX models

<a name="onnx_classification"></a>
#### Classification

|   ONNX Model    | Compression algorithm |Dataset|Accuracy (Drop) %|
| :---: |:---------------------:| :---: | :---: |
|ResNet-50| INT8 (Post-Training)  |ImageNet|74.63 (0.21)|
|ShuffleNet| INT8 (Post-Training)  |ImageNet|47.25 (0.18)|
|GoogleNet| INT8 (Post-Training)  |ImageNet|66.36 (0.3)|
|SqueezeNet V1.0| INT8 (Post-Training)  |ImageNet|54.3 (0.54)|
|MobileNet V2| INT8 (Post-Training)  |ImageNet|71.38 (0.49)|
|DenseNet-121| INT8 (Post-Training)  |ImageNet|60.16 (0.8)|
|VGG-16| INT8 (Post-Training)  |ImageNet|72.02 (0.0)|

<a name="onnx_object_detection"></a>
#### Object Detection

|ONNX Model| Compression algorithm | Dataset |mAP (drop) %|
| :---: |:---------------------:| :---: | :---: |
|SSD1200| INT8 (Post-Training)  |COCO2017|20.17 (0.17)|
|Tiny-YOLOv2| INT8 (Post-Training)  |VOC12|29.03 (0.23)|

## Citing

```
@article{kozlov2020neural,
    title =   {Neural network compression framework for fast model inference},
    author =  {Kozlov, Alexander and Lazarevich, Ivan and Shamporov, Vasily and Lyalyushkin, Nikolay and Gorbachev, Yury},
    journal = {arXiv preprint arXiv:2002.08679},
    year =    {2020}
}
```

## Useful links
- [Documentation](./docs)
- Example scripts (model objects available through links in respective README.md files):
    - [PyTorch](./examples/torch)
    - [TensorFlow](./examples/tensorflow)
- [FAQ](./docs/FAQ.md)
- [Notebooks](https://github.com/openvinotoolkit/openvino_notebooks#-model-training)
- [HuggingFace Optimum Intel](https://huggingface.co/docs/optimum/intel/optimization_ov) utilizes NNCF as a compression backend within the renowned `transformers` repository.
- [Model Optimization Guide](https://docs.openvino.ai/latest/openvino_docs_model_optimization_guide.html)

## Legal Information
[*] Other names and brands may be claimed as the property of others.
