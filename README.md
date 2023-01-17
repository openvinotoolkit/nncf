<div align="center">

# Neural Network Compression Framework (NNCF)

[![GitHub Release](https://img.shields.io/github/v/release/openvinotoolkit/nncf?color=green)](https://github.com/openvinotoolkit/nncf/releases)
[![Website](https://img.shields.io/website?up_color=blue&up_message=docs&url=https%3A%2F%2Fdocs.openvino.ai%2Flatest%2Fopenvino_docs_model_optimization_guide.html)](https://docs.openvino.ai/latest/openvino_docs_model_optimization_guide.html)
[![Apache License Version 2.0](https://img.shields.io/badge/license-Apache_2.0-green.svg)](LICENSE)
[![PyPI Downloads](https://static.pepy.tech/badge/nncf)](https://pypi.org/project/nncf/)
 
</div>

_For the installation instructions, [click here](#installation)._

NNCF provides a suite of advanced algorithms for reducing inference time of Neural Networks executed via [OpenVINO&trade;](https://github.com/openvinotoolkit/openvino) or ONNXRuntime utilizing [OpenVINOExecutionProvider](https://onnxruntime.ai/docs/execution-providers/OpenVINO-ExecutionProvider.html) with attention to models accuracy.

NNCF supports Compression-Aware training and Post-Training Optimization for the models from [PyTorch](https://pytorch.org/), [TensorFlow](https://www.tensorflow.org/), [ONNX](https://onnx.ai/) and [OpenVINO&trade;](https://github.com/openvinotoolkit/openvino).

## Compression-Aware Training Algorithms
Compression-Aware Training Algorithms is a set of algorithms which takes the full training dataset to fine-tune the model after the optimization.
 
|Compression algorithm|PyTorch|TensorFlow|
| :--- | :---: | :---: |
|[Quantization](./docs/compression_algorithms/Quantization.md) | Supported | Supported |
|[Mixed-Precision Quantization](./docs/compression_algorithms/Quantization.md#mixed_precision_quantization) | Supported | Not supported |
|[Binarization](./docs/compression_algorithms/Binarization.md) | Supported | Not supported |
|[Sparsity](./docs/compression_algorithms/Sparsity.md) | Supported | Supported |
|[Filter pruning](./docs/compression_algorithms/Pruning.md) | Supported | Supported |

All of these algorithms can be executed through [Accuracy-Aware model training](./docs/Usage.md#accuracy-aware-model-training) pipelines. 
They allow to achieve the maximum compression level or to reduce the fine-tuning time, while staying within the range of the user-defined maximum accuracy degradation.

## Post-Training Optimization Algorithms
Post-Training Optimization Algorithms is a set of algorithms which takes the subset of validation dataset and utilizes it for calibration of optimization parameters. 


| Compression algorithm                                                       |         ONNX          |TensorFlow|         PyTorch          |OpenVINO|
|:----------------------------------------------------------------------------|:------------------------:| :---: |:------------------------:| :---: |
| [Quantization](./docs/compression_algorithms/post_training/Quantization.md) | Supported  | Supported (Experimental)| Supported (Experimental) | Supported (Experimental)|

## Usage of Compression-Aware Training Algorithms

The basic workflow steps:
1) Load an optimization configuration.
2) Pass the original model along with the optimization configuration to the `create_compressed_model` function. This function returns (1) a model with additional modifications necessary to enable algorithm-specific compression during fine-tuning and (2) a handle to the object allowing you to control the compression during the training process.
3) Fine-tune the modified model.

NNCF provides [samples](#compression-aware-training) and [notebooks](#model-compression-notebooks) that demonstrate the big spectrum of Compression-Aware Training Algorithms usage scenarios for PyTorch and TensorFlow. 

[Compression results](#nncf-compressed-model-zoo) achievable with the NNCF-powered samples can be found in a table at 
the end of this document.

### PyTorch

```python
import torch
import nncf  # Important - must be imported before any other external package that depends on torch

from nncf import NNCFConfig
from nncf.torch import create_compressed_model, register_default_init_args

# Instantiate your uncompressed model
from torchvision.models.resnet import resnet50
model = resnet50()
# Provide data loaders for compression algorithm initialization, if necessary
import torchvision.datasets as datasets
representative_dataset = datasets.ImageFolder("/path")
init_loader = torch.utils.data.DataLoader(representative_dataset)

# Step 1: Load a configuration file to specify compression
nncf_config = NNCFConfig.from_json("resnet50_int8.json")
nncf_config = register_default_init_args(nncf_config, init_loader)

# Step 2: Apply the specified compression algorithms to the model
compression_ctrl, compressed_model = create_compressed_model(model, nncf_config)

# Step 3: Fine-tune
# Now use compressed_model as a usual torch.nn.Module 
# to fine-tune compression parameters along with the model weights

# ... the rest of the usual PyTorch-powered training pipeline

# Export to ONNX or .pth when done fine-tuning
compression_ctrl.export_model("compressed_model.onnx")
torch.save(compressed_model.state_dict(), "compressed_model.pth")
```

**NOTE (PyTorch)**: Due to the way NNCF works within the PyTorch backend, `import nncf` must be done before any other import of `torch` in your package _or_ in third-party packages that your code utilizes, otherwise the compression may be applied incompletely.


### TensorFlow
```python
import tensorflow as tf

from nncf import NNCFConfig
from nncf.tensorflow import create_compressed_model, register_default_init_args

# Instantiate your uncompressed model
from tensorflow.keras.applications import ResNet50
model = ResNet50()
# Provide dataset for compression algorithm initialization
representative_dataset = tf.data.Dataset.list_files("/path/*.jpeg")
# Step 1: Load a configuration file to specify compression
nncf_config = NNCFConfig.from_json("resnet50_int8.json")
nncf_config = register_default_init_args(nncf_config, representative_dataset)

# Step 2: Apply the specified compression algorithms to the model
compression_ctrl, compressed_model = create_compressed_model(model, nncf_config)

# Step 3: Fine-tune
# Now use compressed_model as a usual Keras model
# to fine-tune compression parameters along with the model weights

# ... the rest of the usual TensorFlow-powered training pipeline

# Export to Frozen Graph, TensorFlow SavedModel or .h5  when done fine-tuning 
compression_ctrl.export_model("compressed_model.pb", save_format='frozen_graph')
```

For a more detailed description of NNCF usage in your training code, see [this tutorial](docs/Usage.md). 
For in-depth examples of NNCF integration, browse the [sample scripts](#model-compression-samples) code, or the [example patches](#third-party-repository-integration) to third-party repositories.
For FAQ, visit this [link](./docs/FAQ.md).


## Usage of Post-Training Quantization

NNCF provides [samples](#post-training-quantization) that demonstrate Post-Training Quantization usage for PyTorch, TensorFlow, ONNX, OpenVINO.

To start the algorithm the user should provide:
1) Original model.
2) Validation part of the dataset.
3) Data transformation function from original dataset format to the NNCF required format.
Every backend demands the own return value format of transformation function. 
Please, take a look at the full [description](./docs/compression_algorithms/post_training/Quantization.md). 


The basic workflow steps:
1) Create the data transformation function.
2) Initialize NNCF Dataset with the validation dataset and the transformation function.
3) Run the quantization pipeline.

#### ONNX usage example

```python
import onnx
import nncf
import torch
from torchvision import datasets

# Instantiate your uncompressed model
onnx_model = onnx.load_model('/model_path')

# Provide validation part of the dataset
val_dataset = datasets.ImageFolder("/path")
dataset_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1)

# Step 1: Initialize transformation function
# ONNX backend part of NNCF requires that transformation function returns Dict[str, np.ndarray]
input_name = onnx_model.graph.input[0].name
def transform_fn(data_item):
    images, _ = data_item
    return {input_name: images.numpy()}

# Step 2: Initialize NNCF Dataset
calibration_dataset = nncf.Dataset(dataset_loader, transform_fn)

# Step 3: Run the quantization pipeline
quantized_model = nncf.quantize(onnx_model, calibration_dataset)
```

## Model Compression Samples

For a quicker start with NNCF-powered compression, you can also try the sample scripts, each of which provides a basic training pipeline for classification, semantic segmentation and object detection neural network training correspondingly.

To run the samples please refer to the corresponding tutorials:

### Compression-Aware Training
- PyTorch samples:
  - [Image Classification sample](examples/torch/classification/README.md)
  - [Object Detection sample](examples/torch/object_detection/README.md)
  - [Semantic Segmentation sample](examples/torch/semantic_segmentation/README.md)
- TensorFlow samples:
  - [Image Classification sample](examples/tensorflow/classification/README.md)
  - [Object Detection sample](examples/tensorflow/object_detection/README.md)
  - [Instance Segmentation sample](examples/tensorflow/segmentation/README.md)
### Post-Training Quantization
- [PyTorch sample](examples/post_training_quantization/torch/mobilenet_v2/README.md)
- [Tensorflow sample](examples/post_training_quantization/tensorflow/mobilenet_v2/README.md)
- [ONNX sample](examples/post_training_quantization/onnx/mobilenet_v2/README.md)
- [OpenVINO sample](examples/post_training_quantization/openvino/mobilenet_v2/README.md)

## Model Compression Notebooks 

A collection of ready-to-run Jupyter* notebooks are also available to demonstrate how to use NNCF compression algorithms
to optimize models for inference with the OpenVINO Toolkit.
- [Optimizing PyTorch models with NNCF of OpenVINO by 8-bit quantization](https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/302-pytorch-quantization-aware-training)
- [Optimizing TensorFlow models with NNCF of OpenVINO by 8-bit quantization](https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/305-tensorflow-quantization-aware-training)
- [Post-Training Quantization of Pytorch model with NNCF](https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/112-pytorch-post-training-quantization-nncf)

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
  - TensorFlow\* >=2.4.0, <=2.8.2

This repository is tested on Python* 3.8.10, PyTorch* 1.12.1 (NVidia CUDA\* Toolkit 11.6) and TensorFlow* 2.8.2 (NVidia CUDA\* Toolkit 11.2).

## Installation
We suggest to install or use the package in the [Python virtual environment](https://docs.python.org/3/tutorial/venv.html).

If you want to optimize a model from PyTorch, install PyTorch by following [PyTorch installation guide](https://pytorch.org/get-started/locally/#start-locally). 
If you want to optimize a model from TensorFlow, install TensorFlow by following [TensorFlow installation guide](https://www.tensorflow.org/install/).

#### As a package built from a checked-out repository:

Install the package and its dependencies by running the following in the repository root directory:
```
pip install .
```

Note that if you install NNCF in this manner, the backend frameworks supported by NNCF will not be explicitly installed. NNCF will try to work with whatever backend versions you have installed in your Python environment.

If you want to install both NNCF and the supported PyTorch version in one line, you can do this by running:
```
pip install .[torch]
```
For installation of NNCF along with TensorFlow, run:
```
pip install .[tf]
```
For installation of NNCF for ONNX, run:
```bash
pip install .[onnx]
```


_NB_: For launching example scripts in this repository, we recommend setting the `PYTHONPATH` variable to the root of the checked-out repository once the installation is completed.

#### As a PyPI package:

NNCF can be installed as a regular PyPI package via pip:
```
pip install nncf
```
Use the same `pip install` syntax as above to install NNCF along with the backend package versions in one go, i.e. for NNCF with PyTorch, run:
```
pip install nncf[torch]
```
For installation of NNCF along with TensorFlow, run:
```
pip install nncf[tf]
```
For installation of NNCF for ONNX, run:
```bash
pip install nncf[onnx]
```

NNCF is also available via [conda](https://anaconda.org/conda-forge/nncf):
```
conda install -c conda-forge nncf
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
|MobileNet V3 small|INT8|ImageNet|66.94 (0.73)|
|SqueezeNet V1.1|INT8|ImageNet|58.28 (-0.04)|
|SqueezeNet V1.1|INT8 (per-tensor for weights)|ImageNet|58.26 (-0.02)|
|SqueezeNet V1.1|Mixed, 54.7% INT8 / 45.3% INT4|ImageNet|58.9 (-0.66)|
|ResNet-18|XNOR (weights), scale/threshold (activations)|ImageNet|61.63 (8.17)|
|ResNet-18|DoReFa (weights), scale/threshold (activations)|ImageNet|61.61 (8.19)|
|ResNet-18|Filter pruning, 40%, magnitude criterion|ImageNet|69.26 (0.54)|
|ResNet-18|Filter pruning, 40%, geometric median criterion|ImageNet|69.32 (0.48)|
|ResNet-34|Filter pruning, 50%, geometric median criterion + KD|ImageNet|73.11 (0.19)|
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

|Tensorflow Model|Compression algorithm|Dataset|Accuracy (Drop) %|
| :---: | :---: | :---: | :---: |
|Inception V3|INT8 (per-tensor for weights)|ImageNet|78.36 (-0.44)|
|Inception V3|Sparsity 54% (Magnitude)|ImageNet|77.87 (0.03)|
|Inception V3|INT8 (per-tensor for weights) + Sparsity 61% (RB)|ImageNet|77.58 (0.32)|
|MobileNet V2|INT8 (per-tensor for weights)|ImageNet|71.66 (0.19)|
|MobileNet V2|Sparsity 50% (RB)|ImageNet|71.34 (0.51)|
|MobileNet V2|INT8 (per-tensor for weights) + Sparsity 52% (RB)|ImageNet|71.0 (0.85)|
|MobileNet V3 small|INT8 (per-channel, symmetric for weights; per-tensor, asymmetric for activations) |ImageNet|67.75 (0.63)|
|MobileNet V3 small|INT8 (per-channel, symmetric for weights; per-tensor, asymmetric for activations) + Sparsity 42% (RB)|ImageNet|67.59 (0.79)|
|MobileNet V3 large|INT8 (per-channel, symmetric for weights; per-tensor, asymmetric for activations) |ImageNet|75.04 (0.77)|
|MobileNet V3 large|INT8 (per-channel, symmetric for weights; per-tensor, asymmetric for activations) + Sparsity 42% (RB)|ImageNet|75.29 (0.52)|
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
|RetinaNet|INT8 (per-tensor for weights)|COCO2017|33.18 (0.26)|
|RetinaNet|Sparsity 50% (Magnitude)|COCO2017|33.13 (0.31)|
|RetinaNet|Filter Pruning 40%, geometric_median criterion|COCO2017|32.7 (0.74)|
|RetinaNet|Filter Pruning 40%, geometric_median criterion + INT8 (per-tensor for weights)|COCO2017|32.68 (0.76)|
|YOLOv4|INT8 (per-channel, symmetric for weights; per-tensor, asymmetric for activations)|COCO2017|46.30 (0.74)|
|YOLOv4|Sparsity 50% (Magnitude)|COCO2017|46.54 (0.50)|

<a name="tensorflow_instance_segmentation"></a>
#### Instance segmentation

|TensorFlow Model|<img width="110" height="1">Compression algorithm<img width="110" height="1">|Dataset|mAP (drop) %|
| :---: | :---: | :---: | :---: |
|MaskRCNN|INT8 (per-tensor for weights)|COCO2017|bbox: 37.27 (0.06)<br/>segm: 33.54 (0.02)|
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
