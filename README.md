# Neural Network Compression Framework (NNCF)

This repository contains a PyTorch\*-based framework and samples for neural networks compression.
 
The framework is organized as a Python\* package that can be built and used in a standalone mode. 
The framework architecture is unified to make it easy to add different compression methods.
 
The samples demonstrate the usage of compression algorithms for three different use cases on public models and datasets: Image Classification, Object Detection and Semantic Segmentation.
[Compression results](#nncf-compression-results) achievable with the NNCF-powered samples can be found in a table at the end of this document.

## Key Features

- Support of various compression algorithms, applied during a model fine-tuning process to achieve best compression parameters and accuracy:
    - [Quantization](./docs/compression_algorithms/Quantization.md)
    - [Binarization](./docs/compression_algorithms/Binarization.md)
    - [Sparsity](./docs/compression_algorithms/Sparsity.md)
    - [Filter pruning](./docs/compression_algorithms/Pruning.md)
- Automatic, configurable model graph transformation to obtain the compressed model. The source model is wrapped by the custom class and additional compression-specific layers are inserted in the graph.
- Common interface for compression methods
- GPU-accelerated layers for faster compressed model fine-tuning
- Distributed training support
- Configuration file examples for each supported compression algorithm.
- Git patches for prominent third-party repositories ([mmdetection](https://github.com/open-mmlab/mmdetection), [huggingface-transformers](https://github.com/huggingface/transformers)) demonstrating the process of integrating NNCF into custom training pipelines
- Exporting compressed models to ONNX\* checkpoints ready for usage with [OpenVINO&trade; toolkit](https://github.com/openvinotoolkit/).

## Usage
The NNCF is organized as a regular Python package that can be imported in your target training pipeline script.
The basic workflow is loading a JSON configuration script containing NNCF-specific parameters determining the compression to be applied to your model, and then passing your model along with the configuration script to the `nncf.create_compressed_model` function.
This function returns a wrapped model ready for compression fine-tuning, and handle to the object allowing you to control the compression during the training process:

```python
import torch
import nncf  # Important - should be imported directly after torch
from nncf import create_compressed_model, NNCFConfig, register_default_init_args

# Instantiate your uncompressed model
from torchvision.models.resnet import resnet50
model = resnet50()

# Load a configuration file to specify compression
nncf_config = NNCFConfig.from_json("resnet50_int8.json")

# Provide data loaders for compression algorithm initialization, if necessary
nncf_config = register_default_init_args(nncf_config, train_loader, loss_criterion)

# Apply the specified compression algorithms to the model
comp_ctrl, compressed_model = create_compressed_model(model, nncf_config)

# Now use compressed_model as a usual torch.nn.Module to fine-tune compression parameters along with the model weights

# ... the rest of the usual PyTorch-powered training pipeline

# Export to ONNX or .pth when done fine-tuning
comp_ctrl.export_model("compressed_model.onnx")
torch.save(compressed_model.state_dict(), "compressed_model.pth")
```

For a more detailed description of NNCF usage in your training code, see [Usage.md](./docs/Usage.md). For in-depth examples of NNCF integration, browse the [sample scripts](#model-compression-samples) code, or the [example patches](#third-party-repository-integration) to third-party repositories.

For more details about the framework architecture, refer to the [NNCFArchitecture.md](./docs/NNCFArchitecture.md).


## Model Compression Samples

For a quicker start with NNCF-powered compression, you can also try the sample scripts, each of which provides a basic training pipeline for classification, semantic segmentation and object detection neural network training correspondingly.

To run the samples please refer to the corresponding tutorials:
- [Image Classification sample](examples/classification/README.md)
- [Object Detection sample](examples/object_detection/README.md)
- [Semantic Segmentation sample](examples/semantic_segmentation/README.md)

## Third-party repository integration
NNCF may be straightforwardly integrated into training/evaluation pipelines of third-party repositories.
See [third_party_integration](./third_party_integration) for examples of code modifications (Git patches and base commit IDs are provided) that are necessary to integrate NNCF into select repositories.


## System requirements
- Ubuntu\* 16.04 or later (64-bit)
- Python\* 3.6 or later
- NVidia CUDA\* Toolkit 10.2 or later
- PyTorch\* 1.5 or later.

## Installation
We suggest to install or use the package in the [Python virtual environment](https://docs.python.org/3/tutorial/venv.html).


#### As a package built from a checked-out repository:
1) Install the following system dependencies:

`sudo apt-get install python3-dev`

2) Install the package and its dependencies by running the following in the repository root directory:

- For CPU & GPU-powered execution:
`python setup.py install`
- For CPU-only installation
`python setup.py install --cpu-only`

_NB_: For launching example scripts in this repository, we recommend replacing the `install` option above with `develop` and setting the `PYTHONPATH` variable to the root of the checked-out repository.

#### As a PyPI package:
NNCF can be installed as a regular PyPI package via pip:
```
sudo apt install python3-dev
pip install nncf
```

#### As a Docker image
Use one of the Dockerfiles in the [docker](./docker) directory to build an image with an environment already set up and ready for running NNCF [sample scripts](#model-compression-samples).

## Contributing
Refer to the [CONTRIBUTING.md](./CONTRIBUTING.md) file for guidelines on contributions to the NNCF repository.

## NNCF compression results

Achieved using sample scripts and NNCF configuration files provided with this repository. See README.md files for [sample scripts](#model-compression-samples) for links to exact configuration files and final PyTorch checkpoints.

Quick jump to sample type:

[Classification](#classification)

[Object detection](#object-detection)

[Semantic segmentation](#semantic-segmentation)

[Natural language processing (3rd-party training pipelines)](#nlp)

[Object detection (3rd-party training pipelines)](#object-detection-3rd-party)


#### Classification

|Model|Compression algorithm|Dataset|PyTorch FP32 baseline|PyTorch compressed accuracy|
| :---: | :---: | :---: | :---: | :---: |
|ResNet-50|INT8|ImageNet|76.13|76.05|
|ResNet-50|Mixed, 44.8% INT8 / 55.2% INT4|ImageNet|76.13|76.3|
|ResNet-50|INT8 + Sparsity 61% (RB)|ImageNet|76.13|75.22|
|ResNet-50|INT8 + Sparsity 50% (RB)|ImageNet|76.13|75.60|
|ResNet-50|Filter pruning, 30%, magnitude criterion|ImageNet|76.13|75.7|
|ResNet-50|Filter pruning, 30%, geometric median criterion|ImageNet|76.13|75.7|
|Inception V3|INT8|ImageNet|77.32|76.96|
|Inception V3|INT8 + Sparsity 61% (RB)|ImageNet|77.32|77.02|
|MobileNet V2|INT8|ImageNet|71.81|71.34|
|MobileNet V2|Mixed, 46.6% INT8 / 53.4% INT4|ImageNet|71.81|70.89|
|MobileNet V2|INT8 + Sparsity 52% (RB)|ImageNet|71.81|70.99|
|SqueezeNet V1.1|INT8|ImageNet|58.18|58.02|
|SqueezeNet V1.1|Mixed, 54.7% INT8 / 45.3% INT4|ImageNet|58.18|58.85|
|ResNet-18|XNOR (weights), scale/threshold (activations)|ImageNet|69.76|61.59|
|ResNet-18|DoReFa (weights), scale/threshold (activations)|ImageNet|69.76|61.56|
|ResNet-18|Filter pruning, 30%, magnitude criterion|ImageNet|69.76|68.73|
|ResNet-18|Filter pruning, 30%, geometric median criterion|ImageNet|69.76|68.97|
|ResNet-34|Filter pruning, 30%, magnitude criterion|ImageNet|73.31|72.54|
|ResNet-34|Filter pruning, 30%, geometric median criterion|ImageNet|73.31|72.62|

#### Object detection

|Model|Compression algorithm|Dataset|PyTorch FP32 baseline|PyTorch compressed accuracy|
| :---: | :---: | :---: | :---: | :---: |
|SSD300-BN|INT8|VOC12+07|78.28|78.12|
|SSD300-BN|INT8 + Sparsity 70% (Magnitude)|VOC12+07|78.28|77.94|
|SSD512-BN|INT8|VOC12+07|80.26|80.09|
|SSD512-BN|INT8 + Sparsity 70% (Magnitude)|VOC12+07|80.26|79.88|

#### Semantic segmentation

|Model|Compression algorithm|Dataset|PyTorch FP32 baseline|PyTorch compressed accuracy|
| :---: | :---: | :---: | :---: | :---: |
|UNet|INT8|CamVid|71.95|71.66|
|UNet|INT8 + Sparsity 60% (Magnitude)|CamVid|71.95|71.72|
|ICNet|INT8|CamVid|67.89|67.87|
|ICNet|INT8 + Sparsity 60% (Magnitude)|CamVid|67.89|67.24|
|UNet|INT8|Mapillary|56.23|56.12|
|UNet|INT8 + Sparsity 60% (Magnitude)|Mapillary|56.23|56.0|


#### NLP

|Model|Compression algorithm|Dataset|PyTorch FP32 baseline|PyTorch compressed accuracy|
| :---: | :---: | :---: | :---: | :---: |
|BERT-base-chinese|INT8|XNLI|77.68|77.22|
|BERT-large (Whole Word Masking)|INT8|SQuAD v1.1|93.21 (F1)|92.68 (F1)|
|RoBERTa-large|INT8|MNLI|90.6 (matched)|89.25 (matched)|
|DistilBERT-base|INT8|SST-2|91.1|90.3|
|MobileBERT|INT8|SQuAD v1.1|89.98 (F1)|89.4 (F1)|
|GPT-2|INT8|WikiText-2 (raw)|19.73 (perplexity) | 20.9 (perplexity)|


#### Object detection (3rd party)

|Model|Compression algorithm|Dataset|PyTorch FP32 baseline|PyTorch compressed accuracy|
| :---: | :---: | :---: | :---: | :---: |
|RetinaNet-ResNet50-FPN|INT8|COCO2017|35.6 (avg box mAP)|35.3 (avg box mAP)|
|RetinaNet-ResNet50-FPN|INT8 + Sparsity 50%|COCO2017|35.6 (avg box mAP)|34.7 (avg box mAP)|
|RetinaNet-ResNeXt101-64x4d-FPN|INT8|COCO2017|39.6 (avg box mAP)|39.1 (avg box mAP)|


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
