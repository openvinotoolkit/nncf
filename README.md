<div align="center">

# Neural Network Compression Framework (NNCF)

[Key Features](#key-features) •
[Installation](#installation-guide) •
[Documentation](#documentation) •
[Usage](#usage) •
[Tutorials and Samples](#demos-tutorials-and-samples) •
[Third-party integration](#third-party-repository-integration) •
[Model Zoo](./docs/ModelZoo.md)

[![GitHub Release](https://img.shields.io/github/v/release/openvinotoolkit/nncf?color=green)](https://github.com/openvinotoolkit/nncf/releases)
[![Website](https://img.shields.io/website?up_color=blue&up_message=docs&url=https%3A%2F%2Fdocs.openvino.ai%2Flatest%2Fopenvino_docs_model_optimization_guide.html)](https://docs.openvino.ai/latest/openvino_docs_model_optimization_guide.html)
[![Apache License Version 2.0](https://img.shields.io/badge/license-Apache_2.0-green.svg)](LICENSE)
[![PyPI Downloads](https://static.pepy.tech/badge/nncf)](https://pypi.org/project/nncf/)

</div>

Neural Network Compression Framework (NNCF) provides a suite of post-training and training-time algorithms for neural networks inference optimization in [OpenVINO&trade;](https://docs.openvino.ai) with minimal accuracy drop.

NNCF is designed to work with models from [PyTorch](https://pytorch.org/), [TensorFlow](https://www.tensorflow.org/), [ONNX](https://onnx.ai/) and [OpenVINO&trade;](https://docs.openvino.ai/latest/home.html).

NNCF provides [samples](#demos-tutorials-and-samples) that demonstrate the usage of compression algorithms for different use cases and models. See compression results achievable with the NNCF-powered samples at [Model Zoo page](./docs/ModelZoo.md).

The framework is organized as a Python\* package that can be built and used in a standalone mode. The framework
architecture is unified to make it easy to add different compression algorithms for both PyTorch and TensorFlow deep
learning frameworks.

## Key Features

### Post-Training Compression Algorithms

| Compression algorithm                                                       |OpenVINO|PyTorch|   TensorFlow   |     ONNX       |
|:----------------------------------------------------------------------------| :---: | :---: |:--------:|:------------------:|
| [Post-Training Quantization](./docs/compression_algorithms/post_training/Quantization.md) | Supported | Supported |Supported| Supported |
| [Weights Compression](./docs/compression_algorithms/CompressWeights.md) | Supported | Supported |Not supported| Not supported |

### Training-Time Compression Algorithms

|Compression algorithm|PyTorch|TensorFlow|
| :--- | :---: | :---: |
|[Quantization Aware Training](./docs/compression_algorithms/Quantization.md) | Supported | Supported |
|[Mixed-Precision Quantization](./docs/compression_algorithms/Quantization.md#mixed-precision-quantization) | Supported | Not supported |
|[Binarization](./docs/compression_algorithms/Binarization.md) | Supported | Not supported |
|[Sparsity](./docs/compression_algorithms/Sparsity.md) | Supported | Supported |
|[Filter pruning](./docs/compression_algorithms/Pruning.md) | Supported | Supported |
|[Movement pruning](./nncf/experimental/torch/sparsity/movement/MovementSparsity.md) | Experimental | Not supported |

- Automatic, configurable model graph transformation to obtain the compressed model.
  > **NOTE**: Limited support for TensorFlow models. The models created using Sequential or Keras Functional API are only supported.
- Common interface for compression methods.
- GPU-accelerated layers for faster compressed model fine-tuning.
- Distributed training support.
- Git patch for prominent third-party repository ([huggingface-transformers](https://github.com/huggingface/transformers)) demonstrating the process of integrating NNCF into custom training pipelines
- Seamless combination of pruning, sparsity and quantization algorithms. Please refer to [optimum-intel](https://github.com/huggingface/optimum-intel/tree/main/examples/openvino) for examples of
joint (movement) pruning, quantization and distillation (JPQD), end-to-end from NNCF optimization to compressed OpenVINO IR.
- Exporting PyTorch compressed models to ONNX\* checkpoints and TensorFlow compressed models to SavedModel or Frozen Graph format, ready to use with [OpenVINO&trade; toolkit](https://docs.openvino.ai).
- Support for [Accuracy-Aware model training](./docs/Usage.md#accuracy-aware-model-training) pipelines via the [Adaptive Compression Level Training](./docs/accuracy_aware_model_training/AdaptiveCompressionLevelTraining.md) and [Early Exit Training](./docs/accuracy_aware_model_training/EarlyExitTraining.md).

## Documentation

This documentation covers detailed information about NNCF algorithms and functions needed for the contribution to NNCF.

The latest user documentation for NNCF is available [here](https://docs.openvino.ai/latest/openvino_docs_model_optimization_guide.html).

NNCF API documentation can be found [here](https://openvinotoolkit.github.io/nncf/autoapi/nncf/).

## Usage

### Post-Training Quantization

The NNCF PTQ is the simplest way to apply 8-bit quantization. To run the algorithm you only need your model and a small (~300 samples) calibration dataset.

[OpenVINO](https://github.com/openvinotoolkit/openvino) is the preferred backend to run PTQ with, and PyTorch, TensorFlow and ONNX are also supported.

<details open><summary><b>OpenVINO</b></summary>

```python
import nncf
import openvino.runtime as ov
import torch
from torchvision import datasets, transforms

# Instantiate your uncompressed model
model = ov.Core().read_model("/model_path")

# Provide validation part of the dataset to collect statistics needed for the compression algorithm
val_dataset = datasets.ImageFolder("/path", transform=transforms.Compose([transforms.ToTensor()]))
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
val_dataset = datasets.ImageFolder("/path", transform=transforms.Compose([transforms.ToTensor()]))
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
val_dataset = datasets.ImageFolder("/path", transform=transforms.Compose([transforms.ToTensor()]))
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
representative_dataset = datasets.ImageFolder("/path", transform=transforms.Compose([transforms.ToTensor()]))
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

## Demos, Tutorials and Samples

For a quicker start with NNCF-powered compression, try sample notebooks and scripts presented below.

### Jupyter* Notebook Tutorials and Demos

A collection of ready-to-run Jupyter* notebooks tutorials and demos are available to explain and display NNCF compression algorithms for optimizing models for inference with the OpenVINO Toolkit.

| Notebook Tutorial Name                                                                                                                                                                                                                                                                                                                                                                       |                                  Compression Algorithm                                  |  Backend   |               Domain                |
|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------:|:----------:|:-----------------------------------:|
| [BERT Quantization](https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/105-language-quantize-bert)<br>[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openvinotoolkit/openvino_notebooks/blob/main/notebooks/105-language-quantize-bert/105-language-quantize-bert.ipynb)                               |                               Post-Training Quantization                                |  OpenVINO  |                 NLP                 |
| [MONAI Segmentation Model Quantization](https://github.com/openvinotoolkit/openvino_notebooks/blob/main/notebooks/110-ct-segmentation-quantize)<br>[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F110-ct-segmentation-quantize%2F110-ct-scan-live-inference.ipynb)                                 |                               Post-Training Quantization                                |  OpenVINO  |            Segmentation             |
| [PyTorch Model Quantization](https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/112-pytorch-post-training-quantization-nncf)                                                                                                                                                                                                                                          |                               Post-Training Quantization                                |  PyTorch   |        Image Classification         |
| [TensorFlow Model Quantization](https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/301-tensorflow-training-openvino)                                                                                                                                                                                                                                                  |                               Post-Training Quantization                                | Tensorflow |        Image Classification         |
| [Quantization with Accuracy Control](https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/122-quantizing-model-with-accuracy-control)                                                                                                                                                                                                                                   |                    Post-Training Quantization with Accuracy Control                     |  OpenVINO  | Speech-to-Text,<br>Object Detection |
| [PyTorch Training-Time Compression](https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/302-pytorch-quantization-aware-training)                                                                                                                                                                                                                                       |                                Training-Time Compression                                |  PyTorch   |        Image Classification         |
| [TensorFlow Training-Time Compression](https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/301-tensorflow-training-openvino)                                                                                                                                                                                                                                           |                                Training-Time Compression                                | Tensorflow |        Image Classification         |
| [Joint Pruning, Quantization and Distillation for BERT](https://github.com/openvinotoolkit/openvino_notebooks/blob/main/notebooks/116-sparsity-optimization)                                                                                                                                                                                                                                 |                      Joint Pruning, Quantization and Distillation                       |  OpenVINO  |                 NLP                 |

Below is a list of notebooks demonstrating OpenVINO conversion and inference together with NNCF compression for models from various domains.

| Demo Model                                                                                                                                                                                                                                                                                                                                                                                  |               Compression Algorithm               |  Backend  |                                Domain                                |
|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-------------------------------------------------:|:---------:|:--------------------------------------------------------------------:|
| [YOLOv8](https://github.com/openvinotoolkit/openvino_notebooks/blob/main/notebooks/230-yolov8-optimization)<br>[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openvinotoolkit/openvino_notebooks/blob/main/notebooks/230-yolov8-optimization/230-yolov8-object-detection.ipynb)                                              |            Post-Training Quantization             | OpenVINO  |  Object Detection,<br>KeyPoint Detection,<br>Instance Segmentation   |
| [YOLOv7](https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/226-yolov7-optimization)                                                                                                                                                                                                                                                                                 |            Post-Training Quantization             | OpenVINO  |                           Object Detection                           |
| [EfficientSAM](https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/274-efficient-sam)                                                                                                                                                                                                                                                                                 |            Post-Training Quantization             | OpenVINO  |                          Image Segmentation                          |
| [Segment Anything Model](https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/237-segment-anything)                                                                                                                                                                                                                                                                    |            Post-Training Quantization             | OpenVINO  |                          Image Segmentation                          |
| [OneFormer](https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/249-oneformer-segmentation)                                                                                                                                                                                                                                                                           |            Post-Training Quantization             | OpenVINO  |                          Image Segmentation                          |
| [InstructPix2Pix](https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/231-instruct-pix2pix-image-editing)                                                                                                                                                                                                                                                             |            Post-Training Quantization             | OpenVINO  |                            Image-to-Image                            |
| [CLIP](https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/228-clip-zero-shot-image-classification)                                                                                                                                                                                                                                                                   |            Post-Training Quantization             | OpenVINO  |                            Image-to-Text                             |
| [BLIP](https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/233-blip-visual-language-processing)                                                                                                                                                                                                                                                                       |            Post-Training Quantization             | OpenVINO  |                            Image-to-Text                             |
| [Segmind-VegaRT](https://github.com/openvinotoolkit/openvino_notebooks/blob/main/notebooks/248-stable-diffusion-xl/248-segmind-vegart.ipynb)                                                                                                                                                                                                                                                |            Post-Training Quantization             | OpenVINO  |                            Text-to-Image                             |
| [Latent Consistency Model](https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/263-latent-consistency-models-image-generation)                                                                                                                                                                                                                                        |            Post-Training Quantization             | OpenVINO  |                            Text-to-Image                             |
| [Würstchen](https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/265-wuerstchen-image-generation)                                                                                                                                                                                                                                                                      |            Post-Training Quantization             | OpenVINO  |                            Text-to-Image                             |
| [ControlNet QR Code Monster](https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/264-qrcode-monster)                                                                                                                                                                                                                                                                  |            Post-Training Quantization             | OpenVINO  |                            Text-to-Image                             |
| [SDXL-turbo](https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/271-sdxl-turbo)                                                                                                                                                                                                                                                                                      |            Post-Training Quantization             | OpenVINO  |                   Text-to-Image,<br>Image-to-Image                   |
| [DeepFloyd IF](https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/238-deepfloyd-if)                                                                                                                                                                                                                                                                                  | Post-Training Quantization,<br>Weight Compression | OpenVINO  |                   Text-to-Image,<br>Image-to-Image                   |
| [ImageBind](https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/239-image-bind)                                                                                                                                                                                                                                                                                       |            Post-Training Quantization             | OpenVINO  |                        Multi-Modal Retrieval                         |
| [Distil-Whisper](https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/267-distil-whisper-asr)                                                                                                                                                                                                                                                                          |            Post-Training Quantization             | OpenVINO  |                            Speech-to-Text                            |
| [Whisper](https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/227-whisper-subtitles-generation)<br>[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openvinotoolkit/openvino_notebooks/blob/main/notebooks/227-whisper-subtitles-generation/227-whisper-convert.ipynb)                                   |            Post-Training Quantization             | OpenVINO  |                            Speech-to-Text                            |
| [MMS Speech Recognition](https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/255-mms-massively-multilingual-speech)                                                                                                                                                                                                                                                   |            Post-Training Quantization             | OpenVINO  |                            Speech-to-Text                            |
| [Grammar Error Correction](https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/214-grammar-correction)                                                                                                                                                                                                                                                                |            Post-Training Quantization             | OpenVINO  |                       NLP, Grammar Correction                        |
| [LLM Instruction Following](https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/275-llm-question-answering)                                                                                                                                                                                                                                                           |                Weight Compression                 | OpenVINO  |                      NLP, Instruction Following                      |
| [Dolly 2.0](https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/240-dolly-2-instruction-following)                                                                                                                                                                                                                                                                    |                Weight Compression                 | OpenVINO  |                      NLP, Instruction Following                      |
| [Stable-Zephyr-3b](https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/273-stable-zephyr-3b-chatbot)                                                                                                                                                                                                                                                                  |                Weight Compression                 | OpenVINO  |                            NLP, Chat Bot                             |
| [LLM Chat Bots](https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/254-llm-chatbot)                                                                                                                                                                                                                                                                                  |                Weight Compression                 | OpenVINO  |                            NLP, Chat Bot                             |

### Post-Training Quantization Examples

Compact scripts demonstrating quantization and corresponding inference speed boost:

| Example Name                                                                                                                             |              Compression Algorithm               |  Backend   |         Domain         |
|:-----------------------------------------------------------------------------------------------------------------------------------------|:------------------------------------------------:|:----------:|:----------------------:|
| [OpenVINO MobileNetV2](./examples/post_training_quantization/openvino/mobilenet_v2/README.md)                                            |            Post-Training Quantization            |  OpenVINO  |  Image Classification  |
| [OpenVINO YOLOv8](./examples/post_training_quantization/openvino/yolov8/README.md)                                                       |            Post-Training Quantization            |  OpenVINO  |    Object Detection    |
| [OpenVINO YOLOv8 QwAС](./examples/post_training_quantization/openvino/yolov8_quantize_with_accuracy_control/README.md)                   | Post-Training Quantization with Accuracy Control |  OpenVINO  |    Object Detection    |
| [OpenVINO Anomaly Classification](./examples/post_training_quantization/openvino/anomaly_stfpm_quantize_with_accuracy_control/README.md) | Post-Training Quantization with Accuracy Control |  OpenVINO  | Anomaly Classification |
| [PyTorch MobileNetV2](./examples/post_training_quantization/torch/mobilenet_v2/README.md)                                                |            Post-Training Quantization            |  PyTorch   |  Image Classification  |
| [PyTorch SSD](./examples/post_training_quantization/torch/ssd300_vgg16/README.md)                                                        |            Post-Training Quantization            |  PyTorch   |    Object Detection    |
| [TensorFlow MobileNetV2](./examples/post_training_quantization/tensorflow/mobilenet_v2/README.md)                                        |            Post-Training Quantization            | TensorFlow |  Image Classification  |
| [ONNX MobileNetV2](./examples/post_training_quantization/onnx/mobilenet_v2/README.md)                                                    |            Post-Training Quantization            |    ONNX    |  Image Classification  |

### Training-Time Compression Examples

These examples provide full pipelines including compression, training and inference for classification, detection and segmentation tasks.

| Example Name                                                                                               |   Compression Algorithm   |  Backend   |        Domain         |
|:-----------------------------------------------------------------------------------------------------------|:-------------------------:|:----------:|:---------------------:|
| [PyTorch Image Classification](./examples/torch/classification/README.md)                                  | Training-Time Compression |  PyTorch   | Image Classification  |
| [PyTorch Object Detection](./examples/torch/object_detection/README.md)                                    | Training-Time Compression |  PyTorch   |   Object Detection    |
| [PyTorch Semantic Segmentation](./examples/torch/semantic_segmentation/README.md)                          | Training-Time Compression |  PyTorch   | Semantic Segmentation |
| [TensorFlow Image Classification](./examples/tensorflow/classification/README.md)                          | Training-Time Compression | TensorFlow | Image Classification  |
| [TensorFlow Object Detection](./examples/tensorflow/object_detection/README.md)                            | Training-Time Compression | TensorFlow |   Object Detection    |
| [TensorFlow Instance Segmentation](./examples/tensorflow/segmentation/README.md)                           | Training-Time Compression | TensorFlow | Instance Segmentation |

## Third-party repository integration

NNCF may be straightforwardly integrated into training/evaluation pipelines of third-party repositories.

### Used by

- [OpenVINO Training Extensions](https://github.com/openvinotoolkit/training_extensions)

  NNCF is integrated into OpenVINO Training Extensions as model optimization backend. So you can train, optimize and export new models based on the available model templates as well as run exported models with OpenVINO.

- [HuggingFace Optimum Intel](https://huggingface.co/docs/optimum/intel/optimization_ov)

  NNCF is used as a compression backend within the renowned `transformers` repository in HuggingFace Optimum Intel.

## Installation Guide

For detailed installation instructions please refer to the [Installation](./docs/Installation.md) page.

NNCF can be installed as a regular PyPI package via pip:

```bash
pip install nncf
```

If you want to install both NNCF and the supported PyTorch version in one line, you can do this by simply running:

```bash
pip install nncf[torch]
```

Other viable options besides `[torch]` are `[tf]`, `[onnx]` and `[openvino]`.

NNCF is also available via [conda](https://anaconda.org/conda-forge/nncf):

```bash
conda install -c conda-forge nncf
```

### System requirements

- Ubuntu\* 18.04 or later (64-bit)
- Python\* 3.7 or later
- Supported frameworks:
  - PyTorch\* >=2.1, <2.3
  - TensorFlow\* >=2.8.4, <=2.12.1
  - ONNX\* ~=1.13.1
  - OpenVINO\* >=2022.3.0

This repository is tested on Python* 3.8.10, PyTorch* 2.2.1 (NVidia CUDA\* Toolkit 11.8) and TensorFlow* 2.12.1 (NVidia CUDA\* Toolkit 11.8).

## NNCF Compressed Model Zoo

List of models and compression results for them can be found at our [Model Zoo page](./docs/ModelZoo.md).

## Citing

```bi
@article{kozlov2020neural,
    title =   {Neural network compression framework for fast model inference},
    author =  {Kozlov, Alexander and Lazarevich, Ivan and Shamporov, Vasily and Lyalyushkin, Nikolay and Gorbachev, Yury},
    journal = {arXiv preprint arXiv:2002.08679},
    year =    {2020}
}
```

## Contributing Guide

Refer to the [CONTRIBUTING.md](./CONTRIBUTING.md) file for guidelines on contributions to the NNCF repository.

## Useful links

- [Documentation](./docs)
- Example scripts (model objects available through links in respective README.md files):
  - [PyTorch](./examples/torch)
  - [TensorFlow](./examples/tensorflow)
- [FAQ](./docs/FAQ.md)
- [Notebooks](https://github.com/openvinotoolkit/openvino_notebooks#-model-training)
- [HuggingFace Optimum Intel](https://huggingface.co/docs/optimum/intel/optimization_ov)
- [OpenVINO Model Optimization Guide](https://docs.openvino.ai/latest/openvino_docs_model_optimization_guide.html)

## Telemetry

NNCF as part of the OpenVINO™ toolkit collects anonymous usage data for the purpose of improving OpenVINO™ tools.
You can opt-out at any time by running the following command in the Python environment where you have NNCF installed:

`opt_in_out --opt_out`

More information is available at https://docs.openvino.ai/latest/openvino_docs_telemetry_information.html.
